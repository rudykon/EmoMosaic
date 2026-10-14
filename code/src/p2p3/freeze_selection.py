"""Freeze model family, seeds and decision rules before any held-out performance."""
import hashlib
import json
import shutil
import numpy as np
import torch
import yaml
from scipy.optimize import minimize_scalar
from data import ROOT,OUT,load_split,write_json
from metrics import tune_thresholds,metrics,classification
from evaluate import scenarios


def main():
    cfg=yaml.safe_load((ROOT/'src/p2p3/config.yaml').read_text());summary=[]
    for variant in cfg['variants']:
        for seed in cfg['seeds']:
            run=OUT/'runs'/f'{variant}_s{seed}'
            if not (run/'complete.json').exists():raise RuntimeError(f'Incomplete training: {run.name}')
            row=json.loads((run/'best_metrics.json').read_text());summary.append(dict(variant=variant,seed=seed,**row))
    main_runs=[r for r in summary if r['variant']=='A0'];best=max(main_runs,key=lambda r:r['score'])
    y=load_split('valid')['y'];ensemble=[];aux=[]
    for j in range(4):
        pp=[np.load(OUT/'runs'/f'A0_s{s}'/f'valid_{j}.npz') for s in cfg['seeds']]
        ensemble.append(np.mean([p['y'] for p in pp],0));aux.append(np.mean([p['logits'] for p in pp],0))
    thresholds=tune_thresholds(np.tile(y,4),np.concatenate(ensemble),cfg['threshold_step'],cfg['threshold_max'])
    em=[metrics(y,p,thresholds) for p in ensemble]
    joined=np.concatenate(ensemble);labels=(np.sign(np.tile(y,4))+1).astype(int)
    def ordinal_nll(log_temperature):
        from scipy.special import expit
        temp=np.exp(log_temperature);pn=expit((-thresholds[0]-joined)/temp);pp=expit((joined-thresholds[1])/temp)
        probs=np.stack([pn,1-pn-pp,pp],-1)
        return float(-np.log(probs[np.arange(len(labels)),labels].clip(1e-8)).mean())
    temperature=float(np.exp(minimize_scalar(ordinal_nll,bounds=(np.log(.03),np.log(2)),method='bounded').x))
    rules={'decision':'intensity_thresholds','thresholds':thresholds,'ordinal_temperature':temperature,'ensemble_seeds':cfg['seeds'],
           'best_single_seed':best['seed'],'best_single_thresholds':best['thresholds'],
           'feature_version':cfg['feature_version'],'text_revision':cfg['text_revision'],
           'ensemble_validation':em,'ensemble_validation_auxiliary':[classification(y,p.argmax(-1)) for p in aux],
           'selection_scope':'Validation only. A0 family prespecified for reliability and explainability; ablations assess components.',
           'checkpoint_sha256':{},'all_validation_results':summary}
    dest=ROOT/'models/rmnet';dest.mkdir(parents=True,exist_ok=True)
    for seed in cfg['seeds']:
        src=OUT/'runs'/f'A0_s{seed}'/'best.pt';shutil.copyfile(src,dest/f'weights_seed{seed}_fp16.pt')
        rules['checkpoint_sha256'][str(seed)]=hashlib.sha256(src.read_bytes()).hexdigest()
    shutil.copyfile(OUT/'runs'/f'A0_s{best["seed"]}'/'best.pt',dest/'weights_fp16.pt')
    write_json(OUT/'selection.json',rules)
    write_json(dest/'decision_rules.json',{k:v for k,v in rules.items() if k!='all_validation_results'})
    write_json(dest/'eval_missing_configs.json',{'placement_seeds':list(range(201,206)),
               'scenarios':scenarios(),'generator':'missing.py:fixed_mask; ceil(rate*valid_length); half-open intervals',
               'missing_py_sha256':hashlib.sha256((ROOT/'src/p2p3/missing.py').read_bytes()).hexdigest()})
    print('Frozen validation selection. Held-out evaluation is now permitted.')


if __name__=='__main__':main()
