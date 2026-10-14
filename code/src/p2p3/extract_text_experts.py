"""Frozen BERT/Qwen hidden states, resumable without labels or remote inference.

BERT retains all 13 aligned token layers. Qwen retains ragged full-text tokens,
character offsets and BERT offsets for identical local-missing interventions.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch

from project_paths import DATA_ROOT
from data import ROOT, OUT, load_split, load_pickle, unwrap
from improvement_common import EXP, digest, initialize_protocol, save_json
from text_encoder_registry import encoder_specs, load_frozen_text_model, encoder_protocol_path


def split_text(split, bert):
    """Recover special-set raw text by numbered counterpart, then verify tokens."""
    with np.load(OUT/'cache'/f'{split}.npz', allow_pickle=False) as cache:
        d = {k:cache[k] for k in ['ids','token_ids','obs','raw_text','files'] if k in cache.files}
    if 'raw_text' in d:
        texts = d['raw_text'].tolist()
    else:
        family = 'attachment3*' if split == 'attachment3' else 'attachment4*'
        candidates = list((DATA_ROOT / 'source_data').glob(f'{family}/**/unaligned/*.pkl'))
        texts = []
        for name in d['files']:
            suffix = re.search(r'(\d+)\.pkl$', str(name)).group(1)
            group = [p for p in candidates if re.search(r'(\d+)$',p.stem).group(1)==suffix]
            if len(group) != 1:
                raise ValueError(f'Expected unique unaligned counterpart for {name}, found {group}')
            raw = unwrap(load_pickle(group[0]))
            text = np.asarray(raw['raw_text']).reshape(-1)
            if len(text) != 1:
                raise ValueError(f'Expected one raw_text for {name}')
            texts.append(str(text[0]))
    encoded = bert(texts, padding='max_length', truncation=True, max_length=50,
                   return_offsets_mapping=True, return_tensors='np')
    matches = np.all(encoded['input_ids'] == d['token_ids'], axis=1)
    return d, texts, encoded, matches


def audit_texts(bert):
    """Audit all inputs without reading any test labels into selection code."""
    result = {}
    for split in ['train','valid','test','attachment3','attachment4']:
        # NPZ columns are selected lazily; test labels are never materialized.
        d, texts, encoded, matches = split_text(split, bert)
        result[split] = {'n': len(texts), 'exact_token_matches': int(matches.sum()),
                         'mismatched_indices': np.flatnonzero(~matches).tolist(),
                         'unk_token_count':int((d['token_ids']==bert.unk_token_id).sum()),
                         'mismatch_only_at_unk':bool(np.all((encoded['input_ids']==d['token_ids']) | (d['token_ids']==bert.unk_token_id))),
                         'cache_sha256': digest(OUT/'cache'/f'{split}.npz')}
    policy = 'raw_text' if all(v['n']==v['exact_token_matches'] for v in result.values()) else 'decoded_text_bert_all_splits'
    value = {'policy':policy,'splits':result,'labels_used':False,
             'special_policy':'Unaligned file supplies raw_text only; all audio/vision and model inputs remain aligned_50.'}
    save_json(EXP/'text_alignment_audit.json', value)
    return policy


def decode_with_offsets(ids, bert):
    """Track original WordPiece slots even when [UNK] precedes a ##suffix.

    Retokenizing decoded text can change such censored subword slots. Incremental
    decoding maps the supplied slots directly and never fills an unknown word.
    """
    excluded={bert.pad_token_id,bert.cls_token_id,bert.sep_token_id}
    offsets=np.zeros((len(ids),2),np.int32)
    prefix=[]
    text=''
    for position,token in enumerate(ids):
        if int(token) in excluded:
            continue
        prefix.append(int(token))
        new=bert.decode(prefix,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if not new.startswith(text):
            raise ValueError('Non-monotonic tokenizer decoding requires explicit offset mapping')
        start=len(text)
        while start<len(new) and new[start].isspace():
            start+=1
        offsets[position]=[start,len(new)]
        text=new
    return text,offsets


def read_text(split, bert, policy):
    d, texts, enc, matches = split_text(split, bert)
    if policy == 'decoded_text_bert_all_splits':
        # [UNK] is supplied censorship, not removable padding. Preserve it so
        # decoding cannot collapse token positions or restore hidden source words.
        decoded=[decode_with_offsets(ids,bert) for ids in d['token_ids']]
        texts=[v[0] for v in decoded]
        enc={'input_ids':d['token_ids'],'offset_mapping':np.stack([v[1] for v in decoded])}
    return d, texts, enc


def open_output(path, shape):
    if path.exists():
        array = np.load(path, mmap_mode='r+')
        if array.shape != shape or array.dtype != np.float16:
            raise ValueError(f'Incompatible partial cache: {path}')
        return array
    return np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', choices=list(encoder_specs()), required=True)
    ap.add_argument('--splits', default='train,valid')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    protocol = initialize_protocol()
    from transformers import AutoModel, AutoTokenizer
    bert = AutoTokenizer.from_pretrained(ROOT/'models/pretrained/bert', local_files_only=True)
    policy = audit_texts(bert)
    spec=encoder_specs()[args.encoder]
    folder = ROOT/spec['local_dir']
    layers = spec['layers']
    tokenizer = bert if args.encoder=='bert' else AutoTokenizer.from_pretrained(folder, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Float32 on CPUs without native BF16. GPU uses BF16 for Qwen, FP32 for BERT.
    dtype = torch.bfloat16 if args.encoder!='bert' and args.device.startswith('cuda') else torch.float32
    model = load_frozen_text_model(args.encoder,folder,args.device,dtype)
    model.requires_grad_(False)
    out = EXP/'features'/args.encoder
    out.mkdir(parents=True,exist_ok=True)
    weight_hash=(digest(folder/'model.safetensors') if args.encoder in ['bert','qwen'] else
                 {p.name:digest(p) for p in sorted(folder.glob('*.safetensors'))})
    identity = {'encoder':args.encoder,'model':spec['repository'],
                'revision':spec['revision'],
                'weights_sha256':weight_hash,'layers':layers,'text_policy':policy,
                'template':None,'pooling':'deferred_mean_max_min','compute_dtype':str(dtype),'cache_dtype':'float16',
                'frozen':True,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if args.encoder.startswith('qwen35_'):
        import transformers
        identity.update(transformers_version=transformers.__version__,torch_version=torch.__version__,
                        language_backbone_only=True,checkpoint_stage='post-trained',use_cache=False,
                        protocol_amendment_sha256=digest(encoder_protocol_path(args.encoder)))
    manifest_path = out/'encoder.json'
    if manifest_path.exists():
        import json
        if json.loads(manifest_path.read_text()) != identity:
            raise ValueError('Existing encoder identity changed')
    save_json(manifest_path,identity)
    for split in args.splits.split(','):
        base = out/split
        base.mkdir(parents=True,exist_ok=True)
        if (base/'complete.json').exists():
            print(f'Already extracted: {args.encoder}/{split}',flush=True)
            continue
        d,texts,benc = read_text(split,bert,policy)
        if args.encoder=='bert':
            ids = d['token_ids']
            lengths = np.repeat(50,len(ids))
            offsets = benc['offset_mapping']
            ptr = np.arange(len(ids)+1,dtype=np.int64)*50
            shape = (len(ids),len(layers),50,model.config.hidden_size)
        else:
            qenc = tokenizer(texts, truncation=False, return_offsets_mapping=True, add_special_tokens=False)
            ids = qenc['input_ids']
            lengths = np.asarray([len(i) for i in ids],np.int64)
            if np.any(lengths == 0) or lengths.max()>model.config.max_position_embeddings:
                raise ValueError('Empty or over-context raw text requires explicit policy')
            ptr = np.r_[0,lengths.cumsum()]
            offsets = np.concatenate([np.asarray(o,np.int32) for o in qenc['offset_mapping']])
            shape = (int(ptr[-1]),len(layers),model.config.hidden_size)
        np.savez_compressed(base/'alignment.npz',ids=d['ids'],token_ptr=ptr,offsets=offsets,
                            bert_offsets=benc['offset_mapping'],lengths=lengths,
                            bert_observed=d['obs'][:,0],bert_token_ids=d['token_ids'],raw_text=np.asarray(texts))
        hidden = open_output(base/'hidden.npy',shape)
        progress = base/'progress.json'
        first = 0
        if progress.exists():
            import json
            first = json.loads(progress.read_text())['completed_samples']
        started = time.time()
        for s in range(first,len(ids),args.batch_size):
            end = min(s+args.batch_size,len(ids))
            if args.encoder=='bert':
                batch_ids = torch.as_tensor(ids[s:end],dtype=torch.long,device=args.device)
                mask = torch.as_tensor(d['obs'][s:end,0],device=args.device)
                safe = mask.clone()
                safe[~safe.any(1),0] = True
                batch = {'input_ids':batch_ids,'attention_mask':safe,
                         'token_type_ids':torch.zeros_like(batch_ids)}
            else:
                padded = tokenizer.pad({'input_ids':ids[s:end]},return_tensors='pt',padding=True)
                batch = {k:v.to(args.device) for k,v in padded.items()}
            with torch.inference_mode():
                hs = model(**batch,output_hidden_states=True,return_dict=True,
                           **({'use_cache':False} if args.encoder!='bert' else {})).hidden_states
                for j,layer in enumerate(layers):
                    value = hs[layer].float().cpu().numpy()
                    if not np.isfinite(value).all() or np.abs(value).max()>65504:
                        raise ValueError('Hidden values cannot be represented safely in float16')
                    if args.encoder=='bert':
                        hidden[s:end,j] = value*d['obs'][s:end,0,:,None]
                    else:
                        for local,i in enumerate(range(s,end)):
                            hidden[ptr[i]:ptr[i+1],j] = value[local,:lengths[i]]
            hidden.flush()
            save_json(progress,{'completed_samples':end,'total_samples':len(ids),
                               'seconds_this_session':time.time()-started})
            print(f'{args.encoder}/{split}: {end}/{len(ids)} ({time.time()-started:.1f}s)',flush=True)
        save_json(base/'complete.json',{'samples':len(ids),'hidden_shape':list(shape),
                                       'hidden_sha256':digest(base/'hidden.npy'),
                                       'alignment_sha256':digest(base/'alignment.npz'),
                                       'encoder_sha256':digest(manifest_path)})


if __name__=='__main__':
    main()
