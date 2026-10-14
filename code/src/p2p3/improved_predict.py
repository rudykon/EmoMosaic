"""Raw-input probability-fusion interface for validation-frozen P2 candidates.

The configuration determines all decisions. No labels or validation search are
read at inference. Attribution outputs explicitly refer to RM core intensity.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from data import ROOT, MODALITIES, unwrap, raw_arrays, zero_runs
from project_paths import resolve_asset
from model import RMNet
from predict import Predictor
from boosted_classifier import sequence_features
from extract_text_experts import decode_with_offsets
from probe_text_experts import llm_observed
from improvement_common import pool_tokens, fusion, apply_bias, text_observed_fraction, masked


class ImprovedPredictor:
    def __init__(self, config_path, device='cpu'):
        self.path=Path(config_path)
        self.config=json.loads(self.path.read_text())
        if self.config.get('schema_version')!=1:
            raise ValueError('Unsupported improved predictor configuration')
        base = self.config.get('base_dir', '.')
        self.base_dir = ROOT if base == '<PROJECT_ROOT>' or Path(base).is_absolute() else (self.path.parent/base).resolve()
        self.device=torch.device(device)
        self.original=Predictor(model_dir=self.resolve(self.config['raw_input_model_dir']),device=device,
                                cache_dir=self.resolve(self.config['raw_input_cache_dir']),load_models=False)
        self.models=[]
        for path in self.config['core_checkpoints']:
            checkpoint=torch.load(self.resolve(path),map_location='cpu',weights_only=False)
            model=RMNet(checkpoint['config'],checkpoint['variant'])
            model.load_state_dict(checkpoint['state_dict'])
            self.models.append(model.to(self.device).eval())
        if not self.models:
            raise ValueError('At least one core checkpoint is required')
        self.layer_mix=bool(self.models[0].cfg.get('text_layers'))
        if any(bool(m.cfg.get('text_layers'))!=self.layer_mix for m in self.models):
            raise ValueError('Mixed layer-input conventions within core ensemble')
        self.bert=None;self.bert_tokenizer=None;self.expert_model=None;self.expert_tokenizer=None
        self.expert=self.config.get('expert')
        self.head=joblib.load(self.resolve(self.expert['head'])) if self.expert and self.expert['kind']!='bert_lora' else None
        self.adapters=None
        if self.layer_mix:
            stats=json.loads(self.resolve(self.config['layer_norm_stats']).read_text())
            self.layer_mean=np.asarray(stats['mean'],np.float32)[None,:,None,:]
            self.layer_std=np.asarray(stats['std'],np.float32)[None,:,None,:]

    def resolve(self,path):
        return resolve_asset(path) if Path(path).is_absolute() or str(path).startswith('<PROJECT_ROOT>/') else self.base_dir/path

    def load_file(self,path):
        return self.original.load_file(path)

    def _bert_setup(self):
        from transformers import AutoModel,AutoTokenizer
        folder=self.resolve(self.config['bert_model_dir'])
        if self.bert_tokenizer is None:
            self.bert_tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
        if self.bert is None:
            self.bert=AutoModel.from_pretrained(folder,local_files_only=True,use_safetensors=True,
                                               torch_dtype=torch.float32).to(self.device).eval()
            self.bert.requires_grad_(False)

    @torch.inference_mode()
    def _bert_states(self,d):
        self._bert_setup()
        tokens=np.asarray(d['text_bert']);tokens=tokens[None] if tokens.ndim==2 else tokens
        b=torch.as_tensor(tokens,dtype=torch.long,device=self.device)
        observed=(b[:,0]!=0)&(b[:,1]!=0)
        safe=observed.clone();safe[~safe.any(1),0]=True
        hs=self.bert(input_ids=b[:,0],attention_mask=safe,token_type_ids=b[:,2],output_hidden_states=True).hidden_states
        # Training caches store float16 activations even on float32 CPUs.
        return np.stack([(h.float()*observed[...,None]).cpu().numpy().astype(np.float16) for h in hs],axis=1)

    @torch.inference_mode()
    def _expert_features(self,d,prepared,states):
        if self.expert['kind']=='svd':
            return sequence_features(prepared)[:,:self.head['width']]
        if self.expert['encoder']=='bert':
            if states is None:
                states=self._bert_states(d)
            return pool_tokens(states[:,self.expert['layer']],prepared['obs'][:,0],self.expert['pooling'])
        from transformers import AutoModel,AutoTokenizer
        if self.bert_tokenizer is None:
            self.bert_tokenizer=AutoTokenizer.from_pretrained(self.resolve(self.config['bert_model_dir']),local_files_only=True)
        if self.expert['encoder'].startswith('qwen35_'):
            from qwen35_worker import Qwen35Worker
            if self.expert_model is None:
                self.expert_model=Qwen35Worker(self.expert['encoder'],self.resolve(self.expert['model_dir']),
                                               self.device,torch.get_num_threads())
            decoded=[decode_with_offsets(ids,self.bert_tokenizer) for ids in prepared['token_ids']]
            values=self.expert_model.encode([t for t,_ in decoded],self.expert['layer'])
            obs=prepared['obs'][:,0]&(prepared['token_ids']!=100)
            keep=np.stack([llm_observed(off,b_off,o) for off,(_,b_off),o in zip(values['offsets'],decoded,obs)])
            keep&=values['attention_mask'].astype(bool)
            return pool_tokens(values['hidden'],keep,self.expert['pooling'])
        if self.expert_model is None:
            folder=self.resolve(self.expert['model_dir'])
            self.expert_tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
            self.expert_tokenizer.pad_token=self.expert_tokenizer.pad_token or self.expert_tokenizer.eos_token
            self.expert_model=AutoModel.from_pretrained(folder,local_files_only=True,use_safetensors=True,
                                                       torch_dtype=torch.float32).to(self.device).eval()
            self.expert_model.requires_grad_(False)
        decoded=[decode_with_offsets(ids,self.bert_tokenizer) for ids in prepared['token_ids']]
        texts=[t if t else self.expert_tokenizer.eos_token for t,_ in decoded]
        encoded=self.expert_tokenizer(texts,truncation=False,padding=True,add_special_tokens=False,
                                      return_offsets_mapping=True,return_tensors='pt')
        offsets=encoded.pop('offset_mapping').numpy()
        inputs={k:v.to(self.device) for k,v in encoded.items()}
        h=self.expert_model(**inputs,output_hidden_states=True,use_cache=False).hidden_states[self.expert['layer']]
        h=h.float().cpu().numpy().astype(np.float16)
        obs=prepared['obs'][:,0]&(prepared['token_ids']!=100)
        keep=np.stack([llm_observed(off,b_off,o) for off,(_,b_off),o in zip(offsets,decoded,obs)])
        keep&=encoded['attention_mask'].numpy().astype(bool)
        return pool_tokens(h,keep,self.expert['pooling'])

    @torch.inference_mode()
    def predict(self,X,mask=None):
        d=dict(unwrap(X))
        if 'text_bert' not in d:
            raise ValueError('Improved inference requires aligned text_bert token IDs and masks')
        xx,oo,ll=self.original.input_tensors(d)
        obs=oo.cpu().numpy();lengths=ll.cpu().numpy()
        tokens=np.asarray(d['text_bert']);tokens=tokens[None] if tokens.ndim==2 else tokens
        prepared=dict(zip(MODALITIES,[v.cpu().numpy() for v in xx]))
        prepared.update(obs=obs,lengths=lengths,token_ids=tokens[:,0].astype(np.int64))
        if mask is not None:
            mask=np.asarray(mask,dtype=bool)
            if mask.ndim==2:mask=mask[None]
            if mask.shape!=obs.shape:raise ValueError(f'Expected mask {obs.shape}, got {mask.shape}')
        prepared=masked(prepared,mask)
        states=None
        if self.layer_mix:
            states=self._bert_states(d)
            values=np.clip((states.astype(np.float32)-self.layer_mean)/self.layer_std,-5,5)*obs[:,0,None,:,None]
            xx=[torch.as_tensor(values,device=self.device),*xx[1:]]
        mt=torch.as_tensor(mask,device=self.device) if mask is not None else None
        parts=[m(xx,oo,ll,mt) for m in self.models]
        core_y=torch.stack([p['y'].float() for p in parts]).mean(0).cpu().numpy()
        core_p=torch.stack([p['logits'].float().softmax(-1) for p in parts]).mean(0).cpu().numpy()
        config=self.config['fusion']
        p=apply_bias(core_p,config['bias']);intensity=core_y
        weight=np.zeros(len(core_y));reg_weight=np.zeros(len(core_y))
        if self.expert:
            if self.expert['kind']=='bert_lora':
                from lora_text_expert import load_adapter,predict_adapter
                if self.adapters is None:
                    self.adapters=[load_adapter(self.resolve(path),self.resolve(self.config['bert_model_dir']))
                                   for path in self.expert['adapters']]
                original_prepared=dict(prepared,obs=obs)
                outputs=[predict_adapter(m,original_prepared,mask) for m in self.adapters]
                ep=np.mean([r['probabilities'] for r in outputs],axis=0)
                er=np.mean([r['intensity'] for r in outputs],axis=0)
            else:
                features=self._expert_features(d,prepared,states)
            if self.expert['kind']=='svd':
                ep=self.head['classifier'].predict_proba(self.head['svd'].transform(features))
                weight=np.full(len(core_y),config['w0'])
                p=apply_bias(weight[:,None]*ep+(1-weight[:,None])*core_p,config['bias'])
            else:
                if self.expert['kind']!='bert_lora':
                    features=self.head['scaler'].transform(features)
                    if 'svd' in self.head:features=self.head['svd'].transform(features)
                    ep=self.head['classifier'].predict_proba(features)
                    er=np.clip(self.head['ridge'].predict(features),-3,3)
                fraction=text_observed_fraction(prepared)
                p,weight=fusion(ep,core_p,fraction,config['w0'],config['k'],config['bias'])
                reg_weight=self.config.get('regression_w0',0.)*fraction**self.config.get('regression_k',config['k'])
                reg_weight=np.where(fraction>0,reg_weight,0.)
                intensity=reg_weight*er+(1-reg_weight)*core_y
        contributions=torch.stack([p['contributions'].float() for p in parts]).mean(0).cpu().numpy()
        core_interaction=torch.stack([p['y_int'].float() for p in parts]).mean(0).cpu().numpy()
        return {'y':intensity,'probs':p,'polarity':np.asarray(['Negative','Neutral','Positive'])[p.argmax(-1)],
                'core_y':core_y,'auxiliary_probs':core_p,'expert_weight':weight,'regression_expert_weight':reg_weight,
                'core_contributions':contributions,'core_interaction':core_interaction,
                'explanation_target':'rmnet_core_intensity'}


def export_csv(config_path,input_dir,output,device='cpu'):
    predictor=ImprovedPredictor(config_path,device)
    rows=[]
    for path in sorted(Path(input_dir).glob('*.pkl')):
        d=predictor.load_file(path)
        _,obs,lengths=raw_arrays(d)
        p=predictor.predict(d)
        if len(lengths)!=1:raise ValueError('One sample per file required')
        row={'file':path.name,'sample_id':str(d.get('id',path.stem)),
             'pred_intensity':float(p['y'][0]),'pred_polarity':p['polarity'][0],
             'core_intensity':float(p['core_y'][0]),'text_expert_weight':float(p['expert_weight'][0]),
             'regression_text_weight':float(p['regression_expert_weight'][0])}
        for j,name in enumerate(['neg','neu','pos']):row[f'p_{name}']=float(p['probs'][0,j])
        for j,name in enumerate(MODALITIES):
            row[f'miss_{name}_intervals']=json.dumps(zero_runs(~obs[0,j,:lengths[0]]))
            row[f'core_contribution_{name}']=float(p['core_contributions'][0,j])
        row['core_interaction']=float(p['core_interaction'][0])
        row['attribution_target']='RM-Net core intensity'
        rows.append(row)
    if not rows:raise ValueError('No input PKLs')
    target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'Wrote {len(rows)} improved predictions: {target}',flush=True)
