"""Delete masked character spans BEFORE frozen Qwen encoding, on valid only.

This is a fidelity contrast to encode-then-mask, never a tuning input. Supplied
[UNK] tokens remain unknown; their unaligned raw counterparts are never used.
"""
import argparse
import json
import time

import joblib
import numpy as np
import torch

from data import ROOT, load_split
from improvement_common import EXP, save_json, metrics, pool_tokens, views, fusion, text_observed_fraction
from extract_text_experts import read_text
from probe_text_experts import llm_observed
from text_encoder_registry import encoder_specs, load_frozen_text_model


def strict_core_predictions(valid,deleted,original_offsets,observed,batch_size):
    """Re-encode core BERT too, scatter surviving content back to original slots.

    Deleted characters are spaces, so tokenizer offsets stay in the original
    coordinate system. Overlapping subtokens are averaged by character overlap;
    no deleted token or beyond-truncation raw word is restored. AV rows stay fixed.
    """
    from transformers import AutoTokenizer,AutoModel
    from predict import Predictor
    from compare_boosted import rm_predict
    from train import device_data
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/pretrained/bert',local_files_only=True)
    model=AutoModel.from_pretrained(ROOT/'models/pretrained/bert',local_files_only=True,use_safetensors=True).eval()
    model.requires_grad_(False)
    core=Predictor()
    restored=np.zeros_like(valid['text'])
    for start in range(0,len(deleted),batch_size):
        batch=tokenizer(deleted[start:start+batch_size],truncation=False,padding=True,
                        return_offsets_mapping=True,return_tensors='pt')
        offsets=batch.pop('offset_mapping').numpy()
        with torch.inference_mode():
            hidden=model(**batch).last_hidden_state.float().numpy()
        for local,i in enumerate(range(start,min(start+batch_size,len(deleted)))):
            original=original_offsets[i]
            overlap=np.maximum(0,np.minimum(original[:,None,1],offsets[local,None,:,1])-
                                np.maximum(original[:,None,0],offsets[local,None,:,0]))
            restored[i]=(overlap@hidden[local])/overlap.sum(1,keepdims=True).clip(1)
            restored[i,0]=hidden[local,0]
            restored[i,valid['lengths'][i]-1]=hidden[local,int(batch['attention_mask'][local].sum())-1]
    stats=core.stats['text']
    restored=np.clip((restored-np.asarray(stats['mean'],np.float32))/np.asarray(stats['std'],np.float32),-5,5)
    restored*=observed[:,0,:,None]
    d=dict(valid,text=restored,obs=observed)
    return rm_predict(core,device_data(d,'cpu'),observed)


def delete_spans(text, offsets, observed):
    if not np.any(observed):
        return ''
    keep=np.ones(len(text),bool)
    for (start,end),visible in zip(offsets,observed):
        if not visible:
            keep[int(start):int(end)]=False
    # Spaces keep surviving fragments on opposite sides of a deletion separate.
    return ''.join(c if k else ' ' for c,k in zip(text,keep))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--encoder',choices=['bert','bert_svd','qwen','qwen7','qwen35_08','qwen35_2','qwen35_4'],required=True)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--batch-size',type=int,default=16)
    args=ap.parse_args()
    source_encoder='bert' if args.encoder=='bert_svd' else args.encoder
    torch.set_num_threads(args.threads)
    from transformers import AutoModel,AutoTokenizer
    base=EXP/'probes'/args.encoder
    selected=json.loads((base/'selection.json').read_text())['selected']
    head=joblib.load(base/f'{selected["name"]}.joblib')
    bert=AutoTokenizer.from_pretrained(ROOT/'models/pretrained/bert',local_files_only=True)
    policy=json.loads((EXP/'text_alignment_audit.json').read_text())['policy']
    _,texts,enc=read_text('valid',bert,policy)
    valid=load_split('valid');vv=views(valid)
    obs=vv[1]['obs'][:,0]
    deleted=[delete_spans(t,o,k) for t,o,k in zip(texts,enc['offset_mapping'],obs)]
    folder=ROOT/encoder_specs()[source_encoder]['local_dir']
    worker=None
    if source_encoder.startswith('qwen35_'):
        from qwen35_worker import Qwen35Worker
        worker=Qwen35Worker(source_encoder,folder,threads=args.threads)
    else:
        tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
        tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
        model=load_frozen_text_model(source_encoder,folder)
    pooled=[]
    started=time.time()
    for start in range(0,len(deleted),args.batch_size):
        batch=deleted[start:start+args.batch_size]
        nonempty=np.asarray([bool(x.strip()) for x in batch])
        # Empty rows get a computational placeholder but no pooled observations.
        if worker is not None:
            values=worker.encode(batch,selected['layer'])
            h=values['hidden'];keep=values['attention_mask'].astype(bool)&nonempty[:,None]
            keep&=np.stack([llm_observed(off,b_off,visible&(ids!=100))
                            for off,b_off,visible,ids in zip(values['offsets'],enc['offset_mapping'][start:start+len(batch)],
                                                           obs[start:start+len(batch)],valid['token_ids'][start:start+len(batch)])])
        else:
            batch=[x if x.strip() else ('' if source_encoder=='bert' else tokenizer.eos_token) for x in batch]
            inputs=tokenizer(batch,add_special_tokens=source_encoder=='bert',truncation=False,padding=True,return_tensors='pt')
            with torch.inference_mode():
                h=model(**inputs,output_hidden_states=True).hidden_states[selected['layer']].float().numpy()
            keep=inputs['attention_mask'].numpy().astype(bool)&nonempty[:,None]
        if source_encoder=='bert':
            keep[:,0]&=obs[start:start+len(batch),0]
            for local,i in enumerate(range(start,start+len(batch))):
                sep=int(inputs['attention_mask'][local].sum())-1
                keep[local,sep]&=obs[i,valid['lengths'][i]-1]
        # Match the float16 cached-activation precision of encode-then-mask.
        pooled.append(pool_tokens(h.astype(np.float16),keep,selected['pooling']))
        print(f'{args.encoder} strict valid: {min(start+args.batch_size,len(deleted))}/{len(deleted)}',flush=True)
    if worker is not None:worker.close()
    x=head['scaler'].transform(np.concatenate(pooled))
    if 'svd' in head:x=head['svd'].transform(x)
    p=head['classifier'].predict_proba(x);r=np.clip(head['ridge'].predict(x),-3,3)
    with np.load(base/f'{selected["name"]}_valid.npz') as z:
        earlier=metrics(valid['y'],z['probabilities'][1],z['intensity'][1])
    result={'expert':selected,'encode_then_mask':earlier,'delete_then_reencode':metrics(valid['y'],p,r),
            'seconds':time.time()-started,'test_evaluated':False,'selection_used':False,
            'changed_samples':sum(a!=b for a,b in zip(texts,deleted))}
    fusion_path=EXP/'preliminary_fusion'/args.encoder/'selection.json'
    if fusion_path.exists():
        choice=json.loads(fusion_path.read_text())
        with np.load(EXP/'existing/rm_core_valid.npz') as z:
            cp,cr=z['probabilities'][1],z['intensity'][1]
        fraction=text_observed_fraction(vv[1])
        c=choice['classification']
        fp,_=fusion(p,cp,fraction,c['w0'],c['k'],c['bias'])
        w=choice['regression_w0']*fraction**choice['regression_k']
        result['expert_only_reencoded_fusion']=metrics(valid['y'],fp,w*r+(1-w)*cr)
        strict_core=strict_core_predictions(valid,deleted,enc['offset_mapping'],vv[1]['obs'],args.batch_size)
        strict_fp,_=fusion(p,strict_core['aux_probs'],fraction,c['w0'],c['k'],c['bias'])
        strict_y=w*r+(1-w)*strict_core['intensity']
        result['both_text_branches_reencoded_fusion']=metrics(valid['y'],strict_fp,strict_y)
        result['strict_core']=metrics(valid['y'],strict_core['aux_probs'],strict_core['intensity'])
        result['strict_core_alignment']='Reencoded BERT states scattered to surviving original positions using character overlap; original AV rows and missing masks preserved.'
        np.savez_compressed(base/'strict_core_valid.npz',ids=valid['ids'],probabilities=strict_core['aux_probs'],
                            intensity=strict_core['intensity'],fused_probabilities=strict_fp,fused_intensity=strict_y)
    np.savez_compressed(base/'strict_valid.npz',ids=valid['ids'],probabilities=p,intensity=r,deleted_text=np.asarray(deleted))
    save_json(base/'strict_valid.json',result)


if __name__=='__main__':
    main()
