"""Check pinned frozen text loading and padding before the experiment grid."""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from data import ROOT
from improvement_common import EXP, save_json, digest, pool_tokens
from extract_text_experts import read_text
from text_encoder_registry import encoder_specs, load_frozen_text_model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--encoder',choices=['qwen35_08','qwen35_2','qwen35_4'],required=True)
    ap.add_argument('--threads',type=int,default=6)
    args=ap.parse_args();torch.set_num_threads(args.threads)
    started=time.time();spec=encoder_specs()[args.encoder];folder=ROOT/spec['local_dir']
    downloaded=json.loads((EXP/f'{args.encoder}_download_verified.json').read_text())
    assert downloaded['revision']==spec['revision']
    for entry in downloaded['files']:
        assert digest(folder/entry['name'])==entry['sha256']
    bert=AutoTokenizer.from_pretrained(ROOT/'models/pretrained/bert',local_files_only=True)
    _,texts,_=read_text('train',bert,'decoded_text_bert_all_splits')
    with np.load(EXP/'features/qwen/train/alignment.npz') as old:
        np.testing.assert_array_equal(texts,old['raw_text'])
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    assert tokenizer.padding_side=='right'
    model=load_frozen_text_model(args.encoder,folder)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert all(p.device.type!='meta' for p in model.parameters())
    assert all(p.device.type!='meta' for p in model.buffers())
    samples=['good',texts[0],texts[1],'[UNK] sound and a very different movie']
    batch=tokenizer(samples,add_special_tokens=False,padding=True,return_tensors='pt',return_offsets_mapping=True)
    offsets=batch.pop('offset_mapping').numpy()
    with torch.inference_mode():
        output=model(**batch,use_cache=False,output_hidden_states=True)
    assert output.past_key_values is None
    assert len(output.hidden_states)==model.config.num_hidden_layers+1
    errors=[]
    for i,sample in enumerate(samples):
        single=tokenizer(sample,add_special_tokens=False,return_tensors='pt')
        n=single['input_ids'].shape[1]
        with torch.inference_mode():one=model(**single,use_cache=False,output_hidden_states=True)
        for layer in spec['layers']:
            left=output.hidden_states[layer][i,:n].float().numpy()
            right=one.hidden_states[layer][0].float().numpy()
            assert np.isfinite(left).all() and np.abs(left).max()<65504
            # Accumulation order differs across CPU GEMM batch sizes.
            np.testing.assert_allclose(left,right,rtol=2e-3,atol=2e-3)
            errors.append(float(np.max(np.abs(left-right))))
    last=output.hidden_states[-1].float().numpy().astype(np.float16)
    pooled=pool_tokens(last,batch['attention_mask'].numpy().astype(bool))
    assert pooled.shape==(len(samples),3*spec['hidden_size']) and np.isfinite(pooled).all()
    np.testing.assert_array_equal(pool_tokens(last,np.zeros(last.shape[:2],bool)),np.zeros_like(pooled))
    result={'encoder':args.encoder,'revision':spec['revision'],'weights_verified':True,
            'train_text_identical_to_prior_encoder':True,'parameters':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':0,'language_backbone_only':True,'hidden_state_count':len(output.hidden_states),
            'selected_layers':spec['layers'],'hidden_size':spec['hidden_size'],
            'batch_padding_max_abs_error':max(errors),'all_absent_pooling_zero':True,
            'test_labels_read':False,'seconds':time.time()-started}
    save_json(EXP/f'{args.encoder}_model_verification.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
