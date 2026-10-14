"""Train-only LoRA supplement (separate from the frozen-feature mainline).

Implements W x + (alpha/r) B A x on BERT query/value projections. Base weights
are frozen; only low-rank matrices and task heads are saved. No PEFT dependency.
Reference: https://arxiv.org/abs/2106.09685
"""
from __future__ import annotations

import argparse
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data import ROOT,load_split
from improvement_common import EXP,save_json,metrics,digest
from train import validation_masks
from missing import training_mask


class LoRALinear(nn.Module):
    def __init__(self,base,rank=8,alpha=16,dropout=.05):
        super().__init__()
        self.base=base
        self.base.requires_grad_(False)
        self.a=nn.Parameter(torch.empty(rank,base.in_features))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
        self.scale=alpha/rank
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.base(x)+F.linear(F.linear(self.dropout(x),self.a),self.b)*self.scale


class LoRATextExpert(nn.Module):
    def __init__(self,config,model_dir=None):
        super().__init__()
        from transformers import BertModel
        self.config=config
        self.encoder=BertModel.from_pretrained(model_dir or ROOT/'models/pretrained/bert',local_files_only=True,use_safetensors=True)
        self.encoder.requires_grad_(False)
        for block in self.encoder.encoder.layer:
            for name in ['query','value']:
                base=getattr(block.attention.self,name)
                setattr(block.attention.self,name,LoRALinear(base,config['rank'],config['alpha'],config['lora_dropout']))
        width=self.encoder.config.hidden_size
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,128),nn.GELU(),nn.Dropout(.2))
        self.classifier=nn.Linear(128,3)
        self.regressor=nn.Linear(128,1)

    def encode(self,ids,observed):
        safe=observed.clone();safe[~safe.any(1),0]=True
        return self.encoder(input_ids=ids,attention_mask=safe,token_type_ids=torch.zeros_like(ids)).last_hidden_state

    def classify(self,hidden,keep):
        mean=(hidden*keep[...,None]).sum(1)/keep.sum(1,keepdim=True).clamp_min(1)
        z=self.head(mean)
        return {'logits':self.classifier(z),'y':3*torch.tanh(self.regressor(z).squeeze(-1))}

    def forward(self,ids,observed,mask=None):
        hidden=self.encode(ids,observed)
        keep=observed&(ids!=100)
        if mask is not None:keep=keep&mask
        return self.classify(hidden,keep)

    def adapter_state(self):
        return {k:p.detach().cpu().clone() for k,p in self.named_parameters() if p.requires_grad}


def load_adapter(path,model_dir=None):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=LoRATextExpert(ck['config'],model_dir)
    expected={k for k,p in model.named_parameters() if p.requires_grad}
    if set(ck['adapter'])!=expected:raise ValueError('Incomplete or unexpected adapter keys')
    model.load_state_dict(ck['adapter'],strict=False)
    return model.eval()


@torch.inference_mode()
def predict_adapter(model,data,mask=None,batch_size=32):
    model.eval();pieces=[]
    for start in range(0,len(data['lengths']),batch_size):
        ids=torch.as_tensor(data['token_ids'][start:start+batch_size],dtype=torch.long)
        obs=torch.as_tensor(data['obs'][start:start+batch_size,0],dtype=torch.bool)
        mt=None if mask is None else torch.as_tensor(mask[start:start+batch_size,0],dtype=torch.bool)
        out=model(ids,obs,mt)
        pieces.append((out['logits'].softmax(-1).numpy(),out['y'].numpy()))
    return {'probabilities':np.concatenate([x[0] for x in pieces]),'intensity':np.concatenate([x[1] for x in pieces])}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',default='17,29,43,71,101')
    ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    config={'rank':8,'alpha':16,'lora_dropout':.05,'epochs':10,'patience':3,
            'learning_rate':1e-4,'head_learning_rate':3e-4,'weight_decay':.01,'batch_size':32,
            'classification_weight':1.,'regression_weight':.3,'pooling':'mean_last_hidden',
            'backbone':'google-bert/bert-base-uncased','revision':'86b5e0934494bd15c9632b12f734a8a67f723594',
            'targets':['query','value'],'selection':'mean_valid_clean_and_mixed40_macro_f1',
            'method':'Trainable low-rank update; not a frozen-feature expert'}
    protocol=EXP/'lora_protocol.json'
    if protocol.exists():
        import json
        if json.loads(protocol.read_text())['config']!=config:raise ValueError('LoRA protocol differs')
    else:
        save_json(protocol,{'config':config,'seeds':[17,29,43,71,101],'fit_split':'train','selection_split':'valid',
                            'test_policy':'Frozen evaluation only; no test selection',
                            'rationale':'Original statement permits train-only parameter learning. Small adapters fit the submission budget; branch reported separately.',
                            'base_weights_sha256':digest(ROOT/'models/pretrained/bert/model.safetensors')})
    tr,va=load_split('train'),load_split('valid')
    valid_mask=validation_masks(va['lengths'],20260923)[1]
    ids=torch.as_tensor(tr['token_ids'],dtype=torch.long)
    obs=torch.as_tensor(tr['obs'][:,0],dtype=torch.bool)
    y=torch.as_tensor(tr['y']);labels=(y.sign()+1).long()
    weights=1/torch.bincount(labels,minlength=3).float();weights/=weights.mean()
    for seed in [int(x) for x in args.seeds.split(',')]:
        run=EXP/'lora'/f'seed{seed}';run.mkdir(parents=True,exist_ok=True)
        if (run/'complete.json').exists():continue
        if (run/'history.json').exists():raise RuntimeError(f'Incomplete existing run {run}; verify its process first')
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=LoRATextExpert(config)
        adapter=[p for k,p in model.named_parameters() if p.requires_grad and k.startswith('encoder.')]
        heads=[p for k,p in model.named_parameters() if p.requires_grad and not k.startswith('encoder.')]
        opt=torch.optim.AdamW([{'params':adapter,'lr':config['learning_rate']},{'params':heads,'lr':config['head_learning_rate']}],weight_decay=config['weight_decay'])
        total=config['epochs']*math.ceil(len(y)/config['batch_size']);warm=max(1,total//10)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:(step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/(total-warm))))
        history=[];best=-1.;best_epoch=-1;started=time.time()
        save_json(run/'config.json',dict(config,seed=seed,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        for epoch in range(config['epochs']):
            model.train();order=rng.permutation(len(y));losses=[]
            for start in range(0,len(y),config['batch_size']):
                ix=order[start:start+config['batch_size']]
                mask=training_mask(tr['lengths'][ix],rng,epoch,config['epochs'])[:,0]
                keep=obs[ix]&(ids[ix]!=100)
                opt.zero_grad(set_to_none=True)
                hidden=model.encode(ids[ix],obs[ix])
                clean=model.classify(hidden,keep)
                corrupt=model.classify(hidden,keep&torch.as_tensor(mask))
                ce=(F.cross_entropy(clean['logits'],labels[ix],weight=weights)+F.cross_entropy(corrupt['logits'],labels[ix],weight=weights))/2
                reg=(F.l1_loss(clean['y'],y[ix])+F.l1_loss(corrupt['y'],y[ix]))/2
                loss=ce+config['regression_weight']*reg
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite LoRA training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_([*adapter,*heads],1.)
                opt.step();sched.step();losses.append(float(loss.detach()))
            predictions=[predict_adapter(model,va,m) for m in [None,valid_mask]]
            scored=[metrics(va['y'],p['probabilities'],p['intensity']) for p in predictions]
            score=float(np.mean([s['macro_f1'] for s in scored]))
            row={'epoch':epoch+1,'loss':float(np.mean(losses)),'primary_score':score,'clean':scored[0],
                 'mixed40':scored[1],'seconds':time.time()-started}
            history.append(row);save_json(run/'history.json',history)
            if score>best+1e-5:
                best=score;best_epoch=epoch
                torch.save({'config':config,'seed':seed,'epoch':epoch+1,'adapter':model.adapter_state()},run/'best.pt')
                np.savez_compressed(run/'valid.npz',ids=va['ids'],probabilities=[p['probabilities'] for p in predictions],
                                    intensity=[p['intensity'] for p in predictions])
            print(f'LoRA seed={seed} epoch={epoch+1} score={score:.6f} best={best:.6f} elapsed={row["seconds"]:.1f}s',flush=True)
            if epoch+1>=5 and epoch-best_epoch>=config['patience']:break
        save_json(run/'complete.json',{'best_epoch':best_epoch+1,'primary_score':best,'epochs':len(history),
                                       'seconds':time.time()-started,'adapter_sha256':digest(run/'best.pt')})


if __name__=='__main__':
    main()
