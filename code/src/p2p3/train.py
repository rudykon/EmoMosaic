"""Single-GPU experiment. Model/threshold selection reads train + valid ONLY."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import yaml
from data import ROOT,OUT,MODALITIES,load_split,write_json
from model import RMNet
from missing import fixed_mask, training_mask, TYPES
from metrics import metrics,tune_thresholds,classification


def device_data(d, device):
    return ([torch.as_tensor(d[m],dtype=torch.float32,device=device) for m in MODALITIES],
            torch.as_tensor(d['obs'],dtype=torch.bool,device=device),
            torch.as_tensor(d['lengths'],dtype=torch.long,device=device))


@torch.inference_mode()
def infer(model, tensors, mask=None, batch=128, detailed=False):
    model.eval();xx,oo,ll=tensors;outs={}
    keys=['y','logits','w','y_m','y_int','rho','recon_err']
    if detailed:keys += ['z','alpha']
    for s in range(0,len(ll),batch):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=ll.is_cuda):
            p=model([x[s:s+batch] for x in xx],oo[s:s+batch],ll[s:s+batch],
                    mask[s:s+batch] if mask is not None else None)
        for k in keys:outs.setdefault(k,[]).append(p[k].float().cpu().numpy())
    return {k:np.concatenate(v) for k,v in outs.items()}


def validation_masks(lengths,seed):
    masks=[]
    for rate in [.2,.4,.6]:
        # Balanced seven modality subsets across samples; fixed across all experiments.
        mask=np.ones((len(lengths),3,50),bool)
        rng=np.random.default_rng(seed);assignment=rng.permutation(len(lengths))%7
        for j,kind in enumerate(TYPES):
            idx=np.flatnonzero(assignment==j)
            mask[idx]=fixed_mask(lengths[idx],kind,rate,seed+j)
        masks.append(mask)
    return masks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',default='A0');ap.add_argument('--seed',type=int,default=17)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--epochs',type=int);ap.add_argument('--run-dir')
    ap.add_argument('--lambda-class',type=float)
    ap.add_argument('--selection',choices=['legacy','auxiliary_f1'],default='legacy')
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--text-layers',help='Frozen BERT layer cache directory; enables 13 learned scalar weights')
    args=ap.parse_args();cfg=yaml.safe_load((ROOT/'src/p2p3/config.yaml').read_text())
    if args.epochs:cfg['epochs']=args.epochs
    if args.lambda_class is not None:cfg['lambda_class']=args.lambda_class
    if args.selection=='auxiliary_f1':
        cfg['selection_metric']='half_clean_plus_half_mixed40_auxiliary_macro_f1'
    if args.text_layers:
        cfg['text_layers']=13
        cfg['text_layer_cache']=args.text_layers
    run=Path(args.run_dir) if args.run_dir else OUT/'runs'/f'{args.variant}_s{args.seed}'
    run.mkdir(parents=True,exist_ok=True)
    if (run/'complete.json').exists():
        print(f'Already complete: {run}',flush=True)
        return
    if (run/'history.json').exists():
        raise RuntimeError(f'Incomplete existing run {run}; preserve it and choose a new directory')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads);torch.backends.cuda.matmul.allow_tf32=True
    rng=np.random.default_rng(args.seed)
    if args.text_layers:
        from bert_layer_data import load_layer_split
        tr=load_layer_split('train',args.text_layers);va=load_layer_split('valid',args.text_layers)
    else:
        tr=load_split('train');va=load_split('valid')
    xt,ot,lt=device_data(tr,args.device);tv=device_data(va,args.device)
    y=torch.as_tensor(tr['y'],device=args.device);cy=(y.sign()+1).long()
    counts=torch.bincount(cy,minlength=3).float();cw=(1/counts);cw=cw/cw.mean()
    model=RMNet(cfg,args.variant).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    steps=math.ceil(len(y)/cfg['batch_size']);total=cfg['epochs']*steps;warm=max(1,int(total*cfg['warmup_fraction']))
    def schedule(step):
        return (step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,schedule)
    masks=validation_masks(va['lengths'],cfg['validation_mask_seed'])
    write_json(run/'validation_masks.json',{'seed':cfg['validation_mask_seed'],'rates':[.2,.4,.6],
                'masks_observed': [m.astype(int).tolist() for m in masks]})
    vm=[None]+[torch.as_tensor(m,device=args.device) for m in masks]
    best=-1e9;best_epoch=-1;history=[];start=time.time()
    write_json(run/'config.json',dict(cfg,variant=args.variant,seed=args.seed,
                                    parameters=sum(p.numel() for p in model.parameters()),torch_version=torch.__version__))
    print(f'Start {args.variant} seed={args.seed} params={sum(p.numel() for p in model.parameters())}',flush=True)
    for epoch in range(cfg['epochs']):
        model.train();order=rng.permutation(len(y));losses=[]
        for start_i in range(0,len(y),cfg['batch_size']):
            indices=order[start_i:start_i+cfg['batch_size']];idx=torch.as_tensor(indices,device=args.device)
            keep=training_mask(tr['lengths'][indices],rng,epoch,cfg['epochs'],point=args.variant=='A7',clean=args.variant in ('A1','B_ConcatClean'),cfg=cfg)
            mask=torch.as_tensor(keep,device=args.device)
            x=[v[idx] for v in xt];o=ot[idx];l=lt[idx];target=y[idx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=y.is_cuda):
                clean=model(x,o,l);p=model(x,o,l,mask)
                task=F.l1_loss(clean['y'].float(),target)+F.l1_loss(p['y'].float(),target)
                present=p['obs'].any(-1)
                uni=((p['y_m'].float()-target[:,None]).abs()*present).sum()/present.sum().clamp_min(1)
                ce=(F.cross_entropy(p['logits'].float(),cy[idx],weight=cw)+F.cross_entropy(clean['logits'].float(),cy[idx],weight=cw))/2
                rec=torch.zeros((),device=args.device)
                for m in range(3):
                    hidden=o[:,m]&~p['obs'][:,m]
                    target_features=p['text_target'].detach() if m==0 and args.text_layers else x[m]
                    err=F.smooth_l1_loss(p['recon'][m].float(),target_features,reduction='none').mean(-1)
                    rec=rec+(err*hidden).sum()/hidden.sum().clamp_min(1)/3
                con=F.mse_loss(p['z'].float(),clean['z'].detach().float())+F.l1_loss(p['y'].float(),clean['y'].detach().float())
                loss=task+cfg['lambda_unimodal']*uni+cfg['lambda_class']*ce
                if args.variant!='A2':loss=loss+cfg['lambda_reconstruction']*rec
                if args.variant!='A3':loss=loss+cfg['lambda_consistency']*con
                loss=loss+cfg['lambda_interaction']*p['y_int'].abs().mean()
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at epoch {epoch}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad']);opt.step();sched.step()
            losses.append(float(loss.detach()))
        preds=[infer(model,tv,m) for m in vm]
        thresholds=tune_thresholds(np.tile(va['y'],4),np.concatenate([p['y'] for p in preds]),cfg['threshold_step'],cfg['threshold_max'])
        ms=[metrics(va['y'],p['y'],thresholds) for p in preds]
        score=.5*ms[0]['selection_score']+sum(m['selection_score'] for m in ms[1:])/6
        auxiliary=[classification(va['y'],p['logits'].argmax(-1)) for p in preds]
        if args.selection=='auxiliary_f1':
            score=.5*(auxiliary[0]['macro_f1']+auxiliary[2]['macro_f1'])
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),'score':score,'thresholds':thresholds,
             'clean':ms[0],'missing':ms[1:],'auxiliary_classifier':auxiliary,'seconds':time.time()-start}
        history.append(row);write_json(run/'history.json',history)
        if score>best+1e-5:
            best=score;best_epoch=epoch
            state={k:v.detach().cpu().half() if v.is_floating_point() else v.cpu() for k,v in model.state_dict().items()}
            torch.save({'state_dict':state,'config':cfg,'variant':args.variant,'seed':args.seed,
                        'thresholds':thresholds,'epoch':epoch+1,'score':score},run/'best.pt')
            for j,p in enumerate(preds):np.savez_compressed(run/f'valid_{j}.npz',**p)
            write_json(run/'best_metrics.json',row)
        print(f'{args.variant} s{args.seed} e{epoch+1:02} loss={row["loss"]:.4f} score={score:.4f} best={best:.4f} '
              f'clean MAE={ms[0]["mae"]:.4f} F1={ms[0]["macro_f1"]:.4f} elapsed={row["seconds"]:.1f}s',flush=True)
        if epoch-best_epoch>=cfg['patience']:break
    write_json(run/'complete.json',{'variant':args.variant,'seed':args.seed,'best_epoch':best_epoch+1,
               'epochs_run':len(history),'best_score':best,'seconds':time.time()-start})


if __name__=='__main__':main()
