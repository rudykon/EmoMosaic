"""Frozen-ensemble diagnostics, input-pattern replay, fidelity and imputation baselines."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from data import ROOT,OUT,MODALITIES,load_split,write_json
from predict import Predictor
from train import device_data,infer,validation_masks
from metrics import metrics,classes
from missing import fixed_mask,TYPES
from model import RMNet
from explain import shapley,occlusion,gradient_input,rank_fusion


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def tensors_slice(tensors,indices):return ([x[indices] for x in tensors[0]],tensors[1][indices],tensors[2][indices])


@torch.inference_mode()
def ensemble_infer(predictor,tensors,mask=None,batch=96):
    outs={}
    for start in range(0,len(tensors[2]),batch):
        sl=slice(start,start+batch)
        result=predictor.tensor_forward(tensors_slice(tensors,sl),mask[sl] if mask is not None else None)
        for k,v in result.items():outs.setdefault(k,[]).append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in outs.items()}


def diagnostics(p,device):
    rows=[];summaries={}
    for split in ['valid','test']:
        d=load_split(split);t=device_data(d,device);masks=validation_masks(d['lengths'],p.config['validation_mask_seed'])
        clean=ensemble_infer(p,t);missing=ensemble_infer(p,t,torch.as_tensor(masks[1],device=device))
        summaries[split]={}
        for condition,pred in [('clean',clean),('missing',missing)]:
            summaries[split][condition]=metrics(d['y'],pred['y'],p.thresholds)
            np.savez_compressed(OUT/f'{split}_ensemble_{condition}.npz',**pred)
            strata={
                'intensity_neutral':d['y']==0,'intensity_weak':(np.abs(d['y'])>0)&(np.abs(d['y'])<=.5),
                'intensity_middle':(np.abs(d['y'])>.5)&(np.abs(d['y'])<=1.5),'intensity_extreme':np.abs(d['y'])>1.5,
                'length_short':d['lengths']<15,'length_medium':(d['lengths']>=15)&(d['lengths']<=30),'length_long':d['lengths']>30,
                'vision_zero_low':(~d['obs'][:,2]).sum(1)-(50-d['lengths']) < .15*d['lengths'],
                'vision_zero_high':(~d['obs'][:,2]).sum(1)-(50-d['lengths']) >= .15*d['lengths']}
            disagreement=np.abs(pred['y_m'][:,0]-pred['y_m'][:,2]);cut=np.median(disagreement)
            strata.update({'disagreement_low':disagreement<=cut,'disagreement_high':disagreement>cut})
            for low,high in [(0,.2),(.2,.4),(.4,.6),(.6,1.01)]:
                rho=pred['rho'].mean(1);strata[f'mean_missing_{low}_{high}']=(rho>=low)&(rho<high)
            for group,sel in strata.items():
                if sel.any():rows.append({'split':split,'condition':condition,'group':group,'n':int(sel.sum()),
                     **{k:v for k,v in metrics(d['y'][sel],pred['y'][sel],p.thresholds).items() if k!='confusion'}})
        error_rows=[]
        for condition,pred in [('clean',clean),('missing',missing)]:
            idx=np.argsort(-np.abs(pred['y']-d['y']))[:20]
            for i in idx:error_rows.append({'split':split,'condition':condition,'id':d['ids'][i],
                  'raw_text':d['raw_text'][i],'true_y':float(d['y'][i]),'pred_y':float(pred['y'][i]),
                  'absolute_error':float(abs(pred['y'][i]-d['y'][i])),'valid_len':int(d['lengths'][i]),
                  'pred_class':int(classes(pred['y'],p.thresholds)[i]),'true_class':int(np.sign(d['y'][i])+1)})
        write_csv(OUT/f'{split}_failure_cases.csv',error_rows)
    write_csv(OUT/'stratified_errors.csv',rows);write_json(OUT/'ensemble_metrics.json',summaries)


def replay(p,device):
    d=load_split('test');t=device_data(d,device)
    records=json.loads((OUT/'data_audit.json').read_text())['special']['attachment3']
    rows=[];pooled_pred=[];pooled_gold=[]
    for record in records:
        keep=np.ones((len(d['y']),3,50),bool);reference=record['valid_len']
        for n,L in enumerate(d['lengths']):
            for m,mod in enumerate(MODALITIES):
                for start,end in record[f'miss_{mod}_intervals']:
                    s=int(np.floor(start/reference*L));e=min(L,int(np.ceil(end/reference*L)))
                    keep[n,m,s:e]=False
        pred=ensemble_infer(p,t,torch.as_tensor(keep,device=device))['y']
        pooled_pred.append(pred);pooled_gold.append(d['y'])
        rows.append({'file':record['file'],'n':len(pred),**{k:v for k,v in metrics(d['y'],pred,p.thresholds).items() if k!='confusion'}})
    write_csv(OUT/'attachment3_pattern_replay.csv',rows)
    # Cluster bootstrap by original test sample, preserving dependence of 30 replay masks.
    pp=np.stack(pooled_pred);rng=np.random.default_rng(3301);boots=[]
    for _ in range(500):
        idx=rng.integers(len(d['y']),size=len(d['y']))
        boots.append(metrics(np.tile(d['y'][idx],len(records)),pp[:,idx].ravel(),p.thresholds))
    summary={'estimand':'Performance on held-out test under rescaled observed attachment3 zero patterns, not labels or per-file confidence.',
             'bootstrap_unit':'test sample, retaining all 30 masks; 500 resamples; seed=3301',
             'limitations':'Zero patterns include natural and special-token zero frames; replay may add missingness to already empty frames.',
             'pooled':metrics(np.tile(d['y'],len(records)),pp.ravel(),p.thresholds),
             'ci95':{k:np.quantile([b[k] for b in boots],[.025,.975]).tolist() for k in ['accuracy','macro_f1','weighted_f1','mae','corr']}}
    write_json(OUT/'attachment3_replay_summary.json',summary)


def fidelity(p,device):
    d=load_split('valid');rng=np.random.default_rng(3311)
    # Fixed 72-sample stratified validation subset, 24 per true class.
    idx=np.concatenate([rng.choice(np.flatnonzero(np.sign(d['y'])==c),24,replace=False) for c in [-1,0,1]])
    t=device_data(d,device);t=tensors_slice(t,idx);gold=d['y'][idx]
    with torch.inference_mode():base=p.tensor_forward(t)
    phi,values,residual=shapley(p,t);delta=occlusion(p,t);grad=gradient_input(p,t)
    a=base['alpha'].cpu().numpy();dd=delta.cpu().numpy();gg=grad.cpu().numpy();obs=t[1].cpu().numpy()
    score=rank_fusion(a,dd,gg,obs);con=base['contributions'].cpu().numpy();ph=phi.cpu().numpy()
    correlations=[]
    for m in range(3):correlations.append(float(spearmanr(con[:,m],ph[:,m]).statistic))
    rows=[]
    for mode in ['importance','random']:
        order=np.argsort(-score.reshape(len(idx),-1),axis=1) if mode=='importance' else np.argsort(rng.random((len(idx),150)),axis=1)
        for fraction in [0,.1,.2,.4,.6,.8,1.]:
            for operation in ['delete','retain']:
                mask=np.ones((len(idx),150),bool) if operation=='delete' else np.zeros((len(idx),150),bool)
                for n in range(len(idx)):
                    candidates=[j for j in order[n] if obs[n].reshape(-1)[j]];k=int(round(fraction*len(candidates)))
                    mask[n,candidates[:k]]=operation=='retain'
                with torch.inference_mode():y=p.tensor_forward(t,mask.reshape(len(idx),3,50))['y'].cpu().numpy()
                rows.append({'ordering':mode,'operation':operation,'fraction':fraction,
                    **{k:v for k,v in metrics(gold,y,p.thresholds).items() if k!='confusion'},
                    'prediction_change':float(np.abs(y-base['y'].cpu().numpy()).mean())})
    write_csv(OUT/'fidelity_curves.csv',rows)
    np.savez_compressed(OUT/'validation_explanations.npz',indices=idx,ids=d['ids'][idx],shapley=ph,
           contribution=con,alpha=a,occlusion=dd,gradient_input=gg,rank_fusion=score,obs=obs,
           y=base['y'].cpu().numpy(),empty=values[0].cpu().numpy())
    write_json(OUT/'fidelity_summary.json',{'sample_size':len(idx),'sample_indices':idx.tolist(),
        'max_shapley_efficiency_residual':float(residual.abs().max()),'spearman_contribution_shapley':correlations,
        'main_modality_agreement':float((np.abs(con).argmax(1)==np.abs(ph).argmax(1)).mean()),
        'mean_interaction_absolute':float(base['y_int'].abs().mean()),
        'fraction_interaction_below_15pct_for_abs_y_at_least_point2':float((base['y_int'].abs()[base['y'].abs()>=.2] <= .15*base['y'].abs()[base['y'].abs()>=.2]).float().mean())})


def impute_arrays(d,keep,method,stats):
    xs=[np.array(d[m],copy=True) for m in MODALITIES];obs=d['obs'].copy()&keep
    for m,mod in enumerate(MODALITIES):
        for n,L in enumerate(d['lengths']):
            missing=np.flatnonzero(d['obs'][n,m,:L]&~keep[n,m,:L])
            if not len(missing):continue
            known=np.flatnonzero(obs[n,m,:L])
            if method=='zero':xs[m][n,missing]=np.clip(-np.asarray(stats[mod]['mean'])/np.asarray(stats[mod]['std']),-5,5)
            elif method=='mean' or not len(known):xs[m][n,missing]=0
            elif method=='nearest':
                closest=known[np.abs(known[None]-missing[:,None]).argmin(1)];xs[m][n,missing]=xs[m][n,closest]
            elif method=='linear':
                right=np.searchsorted(known,missing).clip(0,len(known)-1)
                left=(right-1).clip(0,len(known)-1)
                # Clamp outside observed range and linearly interpolate every dimension together.
                left=np.where(missing<known[0],0,left)
                right=np.where(missing>known[-1],len(known)-1,right)
                left=np.where(missing>known[-1],len(known)-1,left)
                denominator=known[right]-known[left]
                fraction=np.divide(missing-known[left],denominator,out=np.zeros(len(missing),float),where=denominator!=0)
                xs[m][n,missing]=(1-fraction[:,None])*xs[m][n,known[left]]+fraction[:,None]*xs[m][n,known[right]]
            obs[n,m,missing]=True
    return xs,obs


def imputation(device,only_seed=None,only_split=None):
    stats=json.loads((ROOT/'models/rmnet/norm_stats.json').read_text());rows=[]
    rules=json.loads((OUT/'selection.json').read_text())
    for split in ([only_split] if only_split else ['valid','test']):
        d=load_split(split)
        for seed in ([only_seed] if only_seed else rules['ensemble_seeds']):
            ck=torch.load(OUT/'runs'/f'B_ConcatClean_s{seed}'/'best.pt',map_location='cpu',weights_only=False)
            model=RMNet(ck['config'],ck['variant']).to(device);model.load_state_dict(ck['state_dict']);model.eval()
            for method in ['zero','mean','linear','nearest']:
                for kind in TYPES:
                    for rate in [0,.2,.4,.6,.8,1.]:
                        for placement in range(201,206):
                            keep=fixed_mask(d['lengths'],kind,rate,placement)
                            x,o=impute_arrays(d,keep,method,stats)
                            tensors=([torch.as_tensor(v,device=device) for v in x],torch.as_tensor(o,device=device),torch.as_tensor(d['lengths'],device=device))
                            pred=infer(model,tensors)['y']
                            rows.append({'split':split,'seed':seed,'method':method,'kind':kind,'rate':rate,'placement':placement,
                                        **{k:v for k,v in metrics(d['y'],pred,ck['thresholds']).items() if k!='confusion'}})
                    print('imputation',split,seed,method,kind,flush=True)
        suffix=f'_s{only_seed}' if only_seed is not None else ''
        write_csv(OUT/'analysis/imputation'/f'imputation_{split}{suffix}.csv',[r for r in rows if r['split']==split])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['diagnostics','replay','fidelity','imputation'],required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--seed',type=int)
    ap.add_argument('--split',choices=['valid','test']);args=ap.parse_args();torch.set_num_threads(2)
    if not (OUT/'selection.json').exists():raise RuntimeError('Freeze selection before analysis')
    if args.task=='imputation':imputation(args.device,args.seed,args.split)
    else:
        p=Predictor(device=args.device)
        {'diagnostics':diagnostics,'replay':replay,'fidelity':fidelity}[args.task](p,args.device)


if __name__=='__main__':main()
