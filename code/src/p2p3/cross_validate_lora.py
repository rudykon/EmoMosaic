"""Five video-disjoint train folds, refitting every LoRA parameter from scratch.

Fixed median validation-selected epoch count; no fold holdout early stopping.
This is component stability, not nested CV or a final fusion performance claim.
"""
import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import StratifiedGroupKFold

from data import ROOT,load_split
from lora_text_expert import LoRATextExpert,predict_adapter
from missing import training_mask
from train import validation_masks
from improvement_common import EXP,save_json,metrics,digest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--output',type=Path,
                    help='New run directory; defaults to outputs/reproduced/lora_group_cv_<UTC>. Historical folds are never overwritten.')
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    config=json.loads((EXP/'lora_protocol.json').read_text())['config']
    best_epochs=[json.loads((EXP/f'lora/seed{s}/complete.json').read_text())['best_epoch'] for s in [17,29,43,71,101]]
    epochs=int(np.median(best_epochs));seed=17
    out=(args.output or ROOT/'outputs/reproduced'/('lora_group_cv_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))).resolve()
    if out == (EXP/'lora/group_cv').resolve() or out.is_relative_to((EXP/'lora/group_cv').resolve()):
        ap.error('Choose a new output directory; the original grouped-CV evidence must be preserved.')
    out.mkdir(parents=True,exist_ok=True)
    protocol={'epochs':epochs,'epoch_rule':'median of the five validation-selected best epochs',
              'seed':seed,'config':config,'fit_split':'train_folds','holdout_used_for_selection':False,
              'scope':'single LoRA component refit per fold; not five-adapter ensemble or RM fusion',
              'interpretation':'Conditional on validation-chosen training duration; not nested unbiased generalization'}
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:
        raise RuntimeError('Preserve old CV before changing protocol')
    save_json(out/'protocol.json',protocol)
    d=load_split('train');y=d['y'];labels=np.sign(y).astype(int)+1
    groups=np.array([str(x).split('$_$')[0] for x in d['ids']])
    masks=validation_masks(d['lengths'],20260923)[1]
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20260926)
    probability=np.zeros((2,len(y),3));intensity=np.zeros((2,len(y)));fold_ids=np.full(len(y),-1)
    rows=[]
    for fold,(fit,held) in enumerate(splitter.split(np.zeros(len(y)),labels,groups)):
        assert not set(groups[fit])&set(groups[held])
        record=out/f'fold{fold}.json';predictions=out/f'fold{fold}.npz';checkpoint=out/f'fold{fold}_adapter.pt'
        if record.exists():
            row=json.loads(record.read_text())
            if not checkpoint.is_file() or digest(checkpoint)!=row.get('adapter_sha256'):
                raise RuntimeError('Saved fold has no matching adapter checkpoint; preserve it and use a new output directory')
            if row['protocol_sha256']!=digest(out/'protocol.json'):raise RuntimeError('CV protocol hash differs')
            with np.load(predictions) as z:
                np.testing.assert_array_equal(z['ids'],d['ids'][held]);p=z['probabilities'];r=z['intensity']
        else:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);rng=np.random.default_rng(seed)
            tr={k:v[fit] for k,v in d.items()};va={k:v[held] for k,v in d.items()}
            ids=torch.as_tensor(tr['token_ids'],dtype=torch.long);obs=torch.as_tensor(tr['obs'][:,0])
            target=torch.as_tensor(tr['y']);classes=(target.sign()+1).long()
            weights=1/torch.bincount(classes,minlength=3).float();weights/=weights.mean()
            model=LoRATextExpert(config)
            adapter=[p for k,p in model.named_parameters() if p.requires_grad and k.startswith('encoder.')]
            heads=[p for k,p in model.named_parameters() if p.requires_grad and not k.startswith('encoder.')]
            opt=torch.optim.AdamW([{'params':adapter,'lr':config['learning_rate']},
                                    {'params':heads,'lr':config['head_learning_rate']}],weight_decay=config['weight_decay'])
            total=config['epochs']*math.ceil(len(fit)/config['batch_size']);warm=max(1,total//10)
            sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:(step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/(total-warm))))
            started=time.time();loss_history=[]
            for epoch in range(epochs):
                model.train();order=rng.permutation(len(fit));losses=[]
                for start in range(0,len(fit),config['batch_size']):
                    ix=order[start:start+config['batch_size']]
                    mask=training_mask(tr['lengths'][ix],rng,epoch,config['epochs'])[:,0]
                    keep=obs[ix]&(ids[ix]!=100);opt.zero_grad(set_to_none=True)
                    hidden=model.encode(ids[ix],obs[ix]);clean=model.classify(hidden,keep)
                    corrupt=model.classify(hidden,keep&torch.as_tensor(mask))
                    loss=(F.cross_entropy(clean['logits'],classes[ix],weight=weights)+F.cross_entropy(corrupt['logits'],classes[ix],weight=weights))/2
                    loss+=config['regression_weight']*(F.l1_loss(clean['y'],target[ix])+F.l1_loss(corrupt['y'],target[ix]))/2
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite CV training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_([*adapter,*heads],1.);opt.step();sched.step()
                    losses.append(float(loss.detach()))
                loss_history.append(float(np.mean(losses)))
                print(f'LoRA fold={fold} epoch={epoch+1}/{epochs} loss={loss_history[-1]:.6f} elapsed={time.time()-started:.1f}',flush=True)
            result=[predict_adapter(model,va,m) for m in [None,masks[held]]]
            p=np.array([v['probabilities'] for v in result]);r=np.array([v['intensity'] for v in result])
            scored=[metrics(va['y'],a,b) for a,b in zip(p,r)]
            row={'fold':fold,'fit_n':len(fit),'holdout_n':len(held),'video_overlap':0,
                 'fit_videos':len(set(groups[fit])),'holdout_videos':len(set(groups[held])),
                 'clean':scored[0],'mixed40':scored[1],'primary_score':float(np.mean([m['macro_f1'] for m in scored])),
                 'seconds':time.time()-started,'loss_history':loss_history,'protocol_sha256':digest(out/'protocol.json')}
            temporary=checkpoint.with_suffix('.pt.tmp')
            torch.save({'config':config,'adapter':model.adapter_state(),'seed':seed,
                        'fold':fold,'epochs':epochs,'protocol_sha256':row['protocol_sha256']},temporary)
            temporary.replace(checkpoint)
            row['adapter_file']=checkpoint.name
            row['adapter_sha256']=digest(checkpoint)
            np.savez_compressed(predictions,ids=d['ids'][held],probabilities=p,intensity=r)
            save_json(record,row);del model,opt,sched
        probability[:,held]=p;intensity[:,held]=r;fold_ids[held]=fold;rows.append(row)
        print(f'LoRA fold={fold}: {row["primary_score"]:.6f}',flush=True)
    assert (fold_ids>=0).all()
    np.savez_compressed(out/'train_group_oof.npz',ids=d['ids'],groups=groups,fold_id=fold_ids,probabilities=probability,intensity=intensity)
    save_json(out/'complete.json',{'protocol':protocol,'folds':rows,'mean_primary':float(np.mean([r['primary_score'] for r in rows])),
                                 'std_primary':float(np.std([r['primary_score'] for r in rows],ddof=1))})


if __name__=='__main__':main()
