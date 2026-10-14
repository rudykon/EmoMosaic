"""Validation-only selection, full-set deletion/insertion and decision robustness."""
import argparse
import itertools
import json
import numpy as np
import pandas as pd
import torch

from data import write_json
from metrics import classes, metrics
from predict import Predictor
from p3_common import (OUT3,SEED,dataset,tensors,eligible,scores,ordering,
                       masked_predictions,corr,provenance)

GRID=np.arange(0,9)/10
METHODS=['occlusion','gradient','alpha','fusion']
SCOPES=['text','audio','vision','joint']


def load_e(split,width):
    z=np.load(OUT3/'explanations'/f'{split}_w{width}.npz')
    seeds={k:z[k] for k in z.files if k not in ['ids','eligible']}
    return {k:v.mean(0) for k,v in seeds.items()},seeds


def ranked_orders(score,obs,scope):
    keep=obs.any(1) if scope==3 else obs[:,scope]
    values=(score*obs).sum(1)/np.maximum(obs.sum(1),1) if scope==3 else score[:,scope]
    return [ordering(s,k) for s,k in zip(values,keep)]


def curve_masks(orders,obs,scope):
    masks=[];ix=[]
    for operation in ['delete','insert']:
        for frac in GRID:
            for n,order in enumerate(orders):
                take=order[:int(np.ceil(frac*len(order)))]
                m=np.ones((3,50),bool)
                targets=range(3) if scope==3 else [scope]
                for j in targets:
                    if operation=='delete':m[j,take]=False
                    else:
                        m[j,obs[n,j]]=False;m[j,take]=True
                masks.append(m);ix.append(n)
    return np.asarray(masks),np.asarray(ix)


def auc_ci(y,pred,base,bootstrap,return_bootstrap=False):
    """Cluster by original sample; all curve steps stay together in bootstrap."""
    delta=np.abs(pred-base[None]);error=np.abs(pred-y[None])
    def auc(a):return np.trapezoid(a,GRID,axis=0)/GRID[-1]
    delta_i=auc(delta);mae_i=auc(error)
    xb=pred[:,bootstrap];yb=y[bootstrap][None]
    xc=xb-xb.mean(-1,keepdims=True);yc=yb-yb.mean(-1,keepdims=True)
    rr=(xc*yc).sum(-1)/np.maximum(np.sqrt((xc*xc).sum(-1)*(yc*yc).sum(-1)),1e-12)
    rauc=np.trapezoid(rr,GRID,axis=0)/GRID[-1]
    rcurve=[float(np.corrcoef(y,row)[0,1]) if np.std(row)>1e-7 else 0. for row in pred]
    out={};draws={}
    for name,point,bs in [('abs_delta',delta_i.mean(),delta_i[bootstrap].mean(1)),
                          ('mae',mae_i.mean(),mae_i[bootstrap].mean(1)),
                          ('pearson',np.trapezoid(rcurve,GRID)/GRID[-1],rauc)]:
        out[name+'_auc']=float(point);out[name+'_ci_low'],out[name+'_ci_high']=map(float,np.quantile(bs,[.025,.975]))
        draws[name]=bs
    return (out,draws) if return_bootstrap else out


def fidelity(p,d):
    out=OUT3/'validation';out.mkdir(parents=True,exist_ok=True)
    obs=eligible(d);t=tensors(d,p.device);y=d['y'];n=len(y)
    rng=np.random.default_rng(SEED);bootstrap=rng.integers(0,n,(500,n))
    summary=[];curves=[];predictions={};selection=[]
    for width in [1,3,5]:
        e,_=load_e('valid',width);base=e['y']
        for method in METHODS:
            score=scores(e,obs,method)
            for scope in range(4):
                key=f'w{width}_{method}_{SCOPES[scope]}'
                path=out/f'curve_{key}.npz'
                if path.exists():pred=np.load(path)['pred']
                else:
                    orders=ranked_orders(score,obs,scope);masks,ix=curve_masks(orders,obs,scope)
                    pred=masked_predictions(p,t,masks,ix).reshape(2,len(GRID),n)
                    np.savez_compressed(path,pred=pred)
                predictions[key]=pred
                for op,j in [('delete',0),('insert',1)]:
                    row=dict(width=width,method=method,scope=SCOPES[scope],operation=op,**auc_ci(y,pred[j],base,bootstrap))
                    summary.append(row)
                    for k,frac in enumerate(GRID):
                        curves.append(dict(width=width,method=method,scope=SCOPES[scope],operation=op,fraction=frac,
                            mae=float(np.abs(pred[j,k]-y).mean()),abs_delta=float(np.abs(pred[j,k]-base).mean()),
                            pearson=float(np.corrcoef(y,pred[j,k])[0,1]) if np.std(pred[j,k])>1e-7 else 0.))
                    if scope==3 and op=='delete':selection.append(row)
            print('fidelity',width,method,flush=True)
    # Random positions: five independent permutations with paired sample bootstrap.
    for scope in range(4):
        key=f'random_{SCOPES[scope]}';path=out/f'curve_{key}.npz'
        if path.exists():pred=np.load(path)['pred']
        else:
            repeats=[]
            for rep in range(5):
                rr=np.random.default_rng(SEED+100+rep)
                orders=ranked_orders(rr.random(obs.shape),obs,scope)
                masks,ix=curve_masks(orders,obs,scope)
                repeats.append(masked_predictions(p,t,masks,ix).reshape(2,len(GRID),n))
            pred=np.stack(repeats);np.savez_compressed(path,pred=pred)
        predictions[key]=pred
        # Each random repetition is evaluated first; interval resamples samples
        # in each repetition then averages metrics, avoiding |mean delta| bias.
        for op,j in [('delete',0),('insert',1)]:
            stats=[auc_ci(y,r[j],base,bootstrap,True) for r in pred]
            avg={k:float(np.mean([r[0][k] for r in stats])) for k in stats[0][0]}
            for metric in ['abs_delta','mae','pearson']:
                draws=np.mean([r[1][metric] for r in stats],axis=0)
                avg[metric+'_ci_low'],avg[metric+'_ci_high']=map(float,np.quantile(draws,[.025,.975]))
            summary.append(dict(width=0,method='random',scope=SCOPES[scope],operation=op,**avg))
            for k,frac in enumerate(GRID):
                curves.append(dict(width=0,method='random',scope=SCOPES[scope],operation=op,fraction=frac,
                    mae=float(np.abs(pred[:,j,k]-y).mean()),abs_delta=float(np.abs(pred[:,j,k]-base).mean()),
                    pearson=float(np.mean([np.corrcoef(y,r[j,k])[0,1] if np.std(r[j,k])>1e-7 else 0. for r in pred]))))
    # Predetermined criterion: greatest joint deletion |prediction change| AUC;
    # ties use smaller width and listed method order. No attachment4 input used.
    chosen=max(selection,key=lambda r:(r['abs_delta_auc'],-r['width'],-METHODS.index(r['method'])))
    width=chosen['width'];method=chosen['method'];e,seeds=load_e('valid',width)
    orders=ranked_orders(scores(e,obs,method),obs,3)
    masks=[];ix=[]
    for k in range(1,6):
        for i,order in enumerate(orders):
            mask=np.ones((3,50),bool);mask[:,order[:k]]=False;masks.append(mask);ix.append(i)
    yp=masked_predictions(p,t,masks,ix).reshape(5,n)
    k_candidates={str(k):float(np.trapezoid(np.r_[np.zeros((1,n)),np.abs(yp[:k]-base)],np.arange(k+1),axis=0).mean()/k) for k in [1,3,5]}
    selected_k=int(max(k_candidates,key=k_candidates.get))
    rule=dict(width=width,method=method,top_k=selected_k,display_top_k=3,
        selection='728 validation only; maximize normalized joint deletion absolute-change AUC over p=0..0.8; k over first 1/3/5 positions by same AUC.',
        selected_value=chosen['abs_delta_auc'],k_candidates=k_candidates,grid=GRID.tolist(),
        bootstrap_replicates=500,random_replicates=5,seed=SEED,
        insertion='Only content positions in the target scope are removed; structural special tokens and other modalities stay fixed.',
        model_assets=provenance())
    write_json(OUT3/'explanation_rules.json',rule)
    # Paired improvement interval, sample bootstrap; positive favors explanation.
    chosen_pred=predictions[f'w{width}_{method}_joint'][0]
    random=predictions['random_joint'][:,0]
    a=np.trapezoid(np.abs(chosen_pred-base),GRID,axis=0)/GRID[-1]
    b=np.trapezoid(np.abs(random-base),GRID,axis=1).mean(0)/GRID[-1]
    diff=a-b
    write_json(out/'paired_fidelity.json',dict(mean_abs_delta_auc_improvement=float(diff.mean()),
        bootstrap_95_ci=np.quantile(diff[bootstrap].mean(1),[.025,.975]).tolist(),n=n,
        note='Selection and evaluation share validation; this interval is descriptive, not an independent confirmatory test.'))
    pd.DataFrame(summary).to_csv(out/'fidelity_auc.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'fidelity_curves.csv',index=False)


def decision(p,d,split):
    rule=json.loads((OUT3/'explanation_rules.json').read_text());e,_=load_e(split,rule['width'])
    obs=eligible(d);t=tensors(d,p.device);score=scores(e,obs,rule['method'])
    n=len(d['ids']);base_class=classes(e['y'],p.thresholds);all_k=[];all_path=[]
    for rep in range(6):
        orders=ranked_orders(score if rep==0 else np.random.default_rng(SEED+200+rep).random(score.shape),obs,3)
        masks=[];ix=[];sizes=[]
        for i,order in enumerate(orders):
            for k in range(1,len(order)+1):
                m=np.ones((3,50),bool);m[:,order[:k]]=False;masks.append(m);ix.append(i)
            sizes.append(len(order))
        pred=masked_predictions(p,t,masks,ix);ks=[];paths=[];start=0
        for i,l in enumerate(sizes):
            yy=pred[start:start+l];start+=l
            flip=np.flatnonzero(classes(yy,p.thresholds)!=base_class[i])
            # l+1 is right-censoring, NOT an observed flip.
            ks.append(int(flip[0]+1) if len(flip) else l+1);paths.append(yy.tolist())
        all_k.append(ks);all_path.append(paths)
    k=np.asarray(all_k);length=obs.any(1).sum(1)
    result=dict(ids=d['ids'].tolist(),flip_k=k[0].tolist(),flip_censored=(k[0]>length).tolist(),
        content_length=length.tolist(),flip_ratio=(k[0]/np.maximum(length,1)).tolist(),
        flip_k_rand=k[1:].mean(0).tolist(),random_k=k[1:].tolist(),
        random_censored=(k[1:]>length).tolist(),prediction_paths=all_path[0],
        definition='First class flip along frozen joint-position ranking. Not a combinatorial global minimum. No flip is censored at length+1. Random mean uses this same censor coding.')
    write_json(OUT3/f'{split}_decision.json',result)


def main():
    from p3_common import require_legacy_protocol
    require_legacy_protocol()
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['fidelity','decision'],default='fidelity')
    ap.add_argument('--split',choices=['valid','attachment4'],default='valid');ap.add_argument('--device',default='cpu')
    args=ap.parse_args();torch.set_num_threads(2);p=Predictor(device=args.device);d=dataset(args.split)
    if args.task=='fidelity':
        assert args.split=='valid';fidelity(p,d)
    else:decision(p,d,args.split)


if __name__=='__main__':main()
