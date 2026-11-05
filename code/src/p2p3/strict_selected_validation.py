"""Validation-only deletion contrast for a selected probability fusion.

Every text encoder sees deleted input. Surviving subtokens are scattered back
by character overlap to the original 50 positions, preserving AV alignment.
Results are diagnostic, never an input to validation configuration search.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from data import ROOT,load_split
from improvement_common import EXP,save_json,digest,masked,metrics,fusion,text_observed_fraction,apply_bias
from improvement_evaluation import CachedEvaluator
from extract_text_experts import read_text
from strict_text_intervention import delete_spans
from train import validation_masks,device_data,infer
from scipy.special import softmax


def scatter_hidden(hidden,new_offsets,old_offsets,ids,lengths,attention,observed):
    restored=np.zeros((len(hidden),50,hidden.shape[-1]),np.float32)
    for i in range(len(hidden)):
        overlap=np.maximum(0,np.minimum(old_offsets[i,:,None,1],new_offsets[i,None,:,1])-
                            np.maximum(old_offsets[i,:,None,0],new_offsets[i,None,:,0]))
        restored[i]=(overlap@hidden[i])/overlap.sum(1,keepdims=True).clip(1)
        for pos in np.flatnonzero(ids[i]==101):restored[i,pos]=hidden[i,0]
        for pos in np.flatnonzero(ids[i]==102):restored[i,pos]=hidden[i,int(attention[i].sum())-1]
    return restored*observed[...,None]


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args();torch.set_num_threads(args.threads)
    from transformers import AutoModel,AutoTokenizer
    from lora_text_expert import load_adapter
    from model import RMNet
    config=json.loads(Path(args.config).read_text());e=config.get('expert')
    identity={'config_sha256':digest(args.config),'method':'all_text_encoders_delete_scatter_v1',
              'source_sha256':digest(__file__)}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    folder=EXP/'strict_selected'/key;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        print(f'Already complete: {folder}');return
    evaluator=CachedEvaluator(args.threads);d=evaluator.dataset('valid')
    mask=validation_masks(d['lengths'],20260923)[1];damaged=masked(d,mask)
    baseline=evaluator.predict(config,'valid',mask)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/pretrained/bert',local_files_only=True)
    _,texts,enc=read_text('valid',tokenizer,'decoded_text_bert_all_splits')
    deleted=[delete_spans(t,o,k) for t,o,k in zip(texts,enc['offset_mapping'],damaged['obs'][:,0])]
    restored_path=EXP/'strict_selected'/'bert_restored_valid.npy'
    # Shared tensor is diagnostic data with a manifest tied to every encoder input.
    input_key=hashlib.sha256((json.dumps(deleted)+digest(ROOT/'models/pretrained/bert/model.safetensors')+
                            digest(__file__)).encode()).hexdigest()
    restored_path=restored_path.with_name(f'bert_restored_{input_key}.npy')
    if restored_path.exists():restored=np.load(restored_path)
    else:
        model=AutoModel.from_pretrained(ROOT/'models/pretrained/bert',local_files_only=True,use_safetensors=True).eval()
        blocks=[]
        for start in range(0,len(d['lengths']),32):
            sl=slice(start,start+32)
            batch=tokenizer(deleted[sl],truncation=False,padding=True,return_offsets_mapping=True,return_tensors='pt')
            offsets=batch.pop('offset_mapping').numpy()
            hidden=model(**batch).last_hidden_state.numpy()
            blocks.append(scatter_hidden(hidden,offsets,enc['offset_mapping'][sl],d['token_ids'][sl],d['lengths'][sl],
                                         batch['attention_mask'].numpy(),damaged['obs'][sl,0]))
        restored=np.concatenate(blocks);np.save(restored_path,restored);del model
    stats=json.loads((ROOT/'models/rmnet/norm_stats.json').read_text())['text']
    text=np.clip((restored-np.asarray(stats['mean'],np.float32))/np.asarray(stats['std'],np.float32),-5,5)
    strict=dict(damaged,text=text*damaged['obs'][:,0,:,None]);core=[]
    for path in config['core_checkpoints']:
        ck=torch.load(evaluator.resolve(config,path),map_location='cpu',weights_only=False)
        if ck['config'].get('text_layers'):raise ValueError('Layer-mix core requires all strict reencoded layers')
        net=RMNet(ck['config'],ck['variant']);net.load_state_dict(ck['state_dict']);net.eval()
        pred=infer(net,device_data(strict,'cpu'),torch.as_tensor(damaged['obs']))
        core.append((softmax(pred['logits'],axis=1),pred['y']))
    cp=np.mean([p[0] for p in core],axis=0);cy=np.mean([p[1] for p in core],axis=0)
    setting=config['fusion'];p=apply_bias(cp,setting['bias']);y=cy
    if e:
        if e['kind']=='bert_lora':
            experts=[]
            for path in e['adapters']:
                model=load_adapter(evaluator.resolve(config,path),evaluator.resolve(config,config['bert_model_dir']))
                parts=[]
                for start in range(0,len(d['lengths']),32):
                    sl=slice(start,start+32)
                    batch=tokenizer(deleted[sl],truncation=False,padding=True,return_offsets_mapping=True,return_tensors='pt')
                    offsets=batch.pop('offset_mapping').numpy()
                    hidden=model.encode(batch['input_ids'],batch['attention_mask'].bool()).numpy()
                    values=scatter_hidden(hidden,offsets,enc['offset_mapping'][sl],d['token_ids'][sl],d['lengths'][sl],
                                          batch['attention_mask'].numpy(),damaged['obs'][sl,0])
                    keep=damaged['obs'][sl,0]&(d['token_ids'][sl]!=100)
                    out=model.classify(torch.as_tensor(values),torch.as_tensor(keep))
                    parts.append((out['logits'].softmax(-1).numpy(),out['y'].numpy()))
                experts.append((np.concatenate([v[0] for v in parts]),np.concatenate([v[1] for v in parts])))
                print(f'Strict LoRA completed: {path}',flush=True)
            ep=np.mean([v[0] for v in experts],axis=0);ey=np.mean([v[1] for v in experts],axis=0)
        elif e['kind']=='linear_text':
            with np.load(evaluator.resolve(config,e['head']).parent/'strict_valid.npz') as z:
                if not np.array_equal(z['ids'],d['ids']):raise ValueError('Strict expert IDs differ')
                ep,ey=z['probabilities'],z['intensity']
        else:
            from boosted_classifier import sequence_features
            head=joblib.load(evaluator.resolve(config,e['head']))
            ep=head['classifier'].predict_proba(head['svd'].transform(sequence_features(strict)[:,:head['width']]))
            ey=None
        if e['kind']=='svd':p=apply_bias(setting['w0']*ep+(1-setting['w0'])*cp,setting['bias'])
        else:
            fraction=text_observed_fraction(damaged)
            p,_=fusion(ep,cp,fraction,setting['w0'],setting['k'],setting['bias'])
            w=np.where(fraction>0,config.get('regression_w0',0)*fraction**config.get('regression_k',setting['k']),0.)
            y=w*ey+(1-w)*cy
    np.savez_compressed(folder/'predictions.npz',ids=d['ids'],probabilities=p,intensity=y,core_probabilities=cp,core_intensity=cy)
    result={'identity':identity,'config':config,'encode_then_mask':metrics(d['y'],baseline['probabilities'],baseline['intensity']),
            'all_text_encoders_delete_reencode':metrics(d['y'],p,y),'strict_core':metrics(d['y'],cp,cy),
            'selection_used':False,'split':'valid','test_evaluated':False,
            'method':'Deletion leaves spaces. Reencoded BERT/LoRA subtokens scatter by character overlap into original positions; AV and masks unchanged.'}
    save_json(folder/'complete.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
