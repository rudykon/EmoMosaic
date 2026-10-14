"""Pre-registered missingness grid. Test evaluation requires frozen selection.json."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from data import ROOT,OUT,MODALITIES,load_split,write_json
from missing import TYPES,fixed_mask
from model import RMNet
from train import device_data,infer
from metrics import metrics


def scenarios(full=True):
    rows=[dict(family='clean',kind='TAV',rate=0.,seed=201)]
    for kind in TYPES:
        for r in np.arange(.1,1.01,.1):
            for seed in range(201,206):rows.append(dict(family='rate',kind=kind,rate=round(float(r),2),seed=seed))
    if not full:return rows
    for kind in TYPES:
        for r in [.3,.5]:
            for location in [0.,.5,1.]:
                for seed in range(201,206):rows.append(dict(family='position',kind=kind,rate=r,location=location,seed=seed))
        for count in [1,2,3,5,8,12]:
            for seed in range(201,206):rows.append(dict(family='duration',kind=kind,count=count,seed=seed))
        for segments in [1,2,4]:
            for seed in range(201,206):rows.append(dict(family='fragment',kind=kind,rate=.4,segments=segments,seed=seed))
        for location in np.linspace(0,1,11):
            # A fixed 3-position window scans relative location without changing duration.
            rows.append(dict(family='scan',kind=kind,count=3,location=round(float(location),2),seed=201))
    for kind in ['TA','TV','AV','TAV']:
        for r in [.2,.4,.6]:
            for colocated in [False,True]:
                for seed in range(201,206):rows.append(dict(family='colocated',kind=kind,rate=r,colocated=colocated,seed=seed))
    for j,row in enumerate(rows):row['scenario_id']=j
    return rows


def scenario_mask(lengths,row):
    kwargs={k:v for k,v in row.items() if k in ('kind','rate','seed','segments','location','count','colocated')}
    return fixed_mask(lengths,**kwargs)


def scalar_metrics(y,pred,thresholds):
    return {k:v for k,v in metrics(y,pred,thresholds).items() if k!='confusion'}


@torch.inference_mode()
def reconstruction_stats(model,tensors,mask,batch=128):
    xx,oo,ll=tensors
    sums=np.zeros((3,4),np.float64) # count, sum target, sum target^2, squared errors
    for start in range(0,len(ll),batch):
        sl=slice(start,start+batch)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=ll.is_cuda):
            p=model([x[sl] for x in xx],oo[sl],ll[sl],mask[sl])
        for m in range(3):
            hidden=oo[sl,m]&~p['obs'][:,m]
            target=xx[m][sl][hidden].double();estimate=p['recon'][m].float()[hidden].double()
            sums[m]+=[target.numel(),float(target.sum()),float(target.square().sum()),float((target-estimate).square().sum())]
    out={}
    for m,name in enumerate('tav'):
        n,s,ss,se=sums[m];den=ss-s*s/max(n,1)
        out[f'recon_r2_{name}']=float(1-se/den) if n and den>1e-8 else ''
        out[f'recon_mse_{name}']=float(se/n) if n else ''
        out[f'recon_n_{name}']=int(n)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--split',choices=['valid','test'],required=True)
    ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    if args.split=='test' and not (OUT/'selection.json').exists():raise RuntimeError('Freeze validation selection before test.')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    run=Path(args.run);checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    variant=checkpoint['variant'];seed=checkpoint['seed'];cfg=checkpoint['config'];thresholds=checkpoint['thresholds']
    model=RMNet(cfg,variant).to(args.device);model.load_state_dict(checkpoint['state_dict']);model.eval()
    d=load_split(args.split);tensors=device_data(d,args.device);y=d['y'];grid=scenarios(full=variant=='A0')
    folder=OUT/'evaluation'/run.name;folder.mkdir(parents=True,exist_ok=True)
    dest=folder/f'{args.split}.csv'
    if (folder/f'{args.split}_complete.json').exists():return
    rows=[];start=time.time()
    for index,sc in enumerate(grid):
        keep=scenario_mask(d['lengths'],sc);mask=torch.as_tensor(keep,device=args.device)
        p=infer(model,tensors,mask,detailed=sc['family']=='clean')
        row=dict(sc,variant=variant,model_seed=seed,split=args.split,n=len(y),**scalar_metrics(y,p['y'],thresholds))
        row['mask_sha256']=hashlib.sha256(keep.tobytes()).hexdigest()
        for m,name in enumerate('tav'):
            row[f'w_{name}']=float(p['w'][:,m].mean());row[f'rho_{name}']=float(p['rho'][:,m].mean())
            present=p['rho'][:,m]<1
            row[f'branch_mae_{name}']=float(np.abs(p['y_m'][present,m]-y[present]).mean()) if present.any() else ''
        row['interaction_absolute']=float(np.abs(p['y_int']).mean())
        row['interaction_fraction_stable']=float((np.abs(p['y_int'])/(np.abs(p['y'])+.1)).mean())
        if variant=='A0' and sc['family']=='duration':row.update(reconstruction_stats(model,tensors,mask))
        rows.append(row)
        if variant=='A0' and sc['family'] in ('duration','position','fragment','colocated'):
            for label,sel in [('short',d['lengths']<15),('medium',(d['lengths']>=15)&(d['lengths']<=30)),('long',d['lengths']>30)]:
                if sel.any():rows.append(dict(sc,variant=variant,model_seed=seed,split=args.split,
                                              stratum=label,n=int(sel.sum()),**scalar_metrics(y[sel],p['y'][sel],thresholds)))
        if sc['family']=='clean':np.savez_compressed(folder/f'{args.split}_clean.npz',**p)
        if variant=='A0' and sc['family']=='rate' and sc['kind']=='TAV' and sc['rate']==.4 and sc['seed']==201:
            details=infer(model,tensors,mask,detailed=True)
            np.savez_compressed(folder/f'{args.split}_missing.npz',**details,mask=keep)
        if index%50==0:print(run.name,args.split,index,len(grid),f'{time.time()-start:.1f}s',flush=True)
    fields=sorted(set(k for r in rows for k in r))
    temp=dest.with_suffix('.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    temp.replace(dest)
    write_json(folder/f'{args.split}_complete.json',{'rows':len(rows),'scenarios':len(grid),'seconds':time.time()-start,
               'checkpoint_sha256':hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),
               'split_cache_sha256':hashlib.sha256((OUT/'cache'/f'{args.split}.npz').read_bytes()).hexdigest()})


if __name__=='__main__':main()
