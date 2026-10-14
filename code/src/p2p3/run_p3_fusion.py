"""Reproducible output explanations for the final frozen fusion model."""
import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch

from project_paths import python_executable
from data import ROOT, write_json
from improvement_common import metrics
from p3_common import OUT3, dataset, eligible, ordering, digest, attachment_paths
from p3_fusion import FusionExplainer, CONFIG, SEED, LABELS, provenance, explain
from model_paths import asset_matches, current_model_paths

GRID = np.arange(9)/10


def fixed_protocol():
    protocol = dict(schema='p3_final_fusion_v1', model_assets=provenance(),
        engine_sha256=digest(ROOT/'src/p2p3/p3_fusion.py'),
        targets=['fused_intensity','fixed_original_predicted_class_probability'],
        local_method='exact_single_position_occlusion', width=1, display_top_k=3,
        ranking=dict(intensity='descending absolute intensity change',
                     classification='descending signed original-class probability decrease'),
        intervention='post_encoding_observation_mask; recompute expert pooling and reliability weights',
        grid=GRID.tolist(), random_replicates=5, bootstrap_replicates=500, seed=SEED,
        fitting='none; no model, threshold, fusion, or ranking hyperparameter fitting',
        classification='argmax of final biased fused probabilities; no intensity thresholds',
        gradient='not used; RM-only gradients cannot explain the additional expert',
        inputs={str(p.relative_to(ROOT)):digest(p) for p in [
            (OUT3/'cache'/f'{s}.npz') if (OUT3/'cache'/f'{s}.npz').exists() else ROOT/'outputs/problem2/cache'/f'{s}.npz'
            for s in ['valid','test','attachment4']]},
        note='Validation reused from model development; descriptive sample bootstrap, not independent confirmation.')
    path=OUT3/'explanation_rules.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != protocol:
            # Only a checked directory migration may reuse existing evidence.
            comparable = dict(previous, model_assets=protocol['model_assets'],
                              engine_sha256=protocol['engine_sha256'])
            same_assets = (set(current_model_paths(previous['model_assets'])) == set(protocol['model_assets'])
                           and all(asset_matches(ROOT/name, value)
                                   for name, value in previous['model_assets'].items()))
            same_engine = asset_matches(ROOT/'src/p2p3/p3_fusion.py', previous['engine_sha256'])
            if comparable != protocol or not same_assets or not same_engine:
                raise RuntimeError('Model/input/protocol changed; archive the previous results before rerunning')
        return previous
    write_json(path,protocol)
    return protocol


def validate_public(p, d):
    records=[]
    for i in [0,9,19]:
        raw=p.public.load_file(attachment_paths()[i])
        masks=np.ones((4,3,50),bool)
        masks[1,0]=False; masks[2,:,3:9]=False; masks[3]=False
        expected=[p.public.predict(raw,m) for m in masks]
        actual=p.forward([i]*4,masks)
        for j,(name,r) in enumerate(zip(['clean','no_text','partial','empty'],expected)):
            dy=float(np.max(abs(r['y']-actual['y'][j])))
            dp=float(np.max(abs(r['probs']-actual['probs'][j])))
            assert dy < 2e-5 and dp < 2e-5,(i,name,dy,dp)
            assert r['probs'].argmax(-1)[0]==actual['probs'][j].argmax()
            records.append(dict(id=str(d['ids'][i]),condition=name,max_intensity_error=dy,max_probability_error=dp))
    write_json(OUT3/'interface_verification.json',dict(passed=True,comparisons=records,
        reference='ImprovedPredictor.predict raw-input path; original deployed CPU adapters'))


def score_order(e, target):
    keep=e['eligible'].any(1)
    if target=='intensity': score=abs(e['occlusion_y'][:,3])
    else:
        c=e['probs'].argmax(1)
        score=e['occlusion_probs'][np.arange(len(c)),3,:,c]
    return [ordering(a,b) for a,b in zip(score,keep)]


def curve(p, orders, name):
    folder=OUT3/'validation'; folder.mkdir(exist_ok=True)
    path=folder/f'curve_{name}.npz'
    if path.exists():
        with np.load(path) as z:return dict(z)
    masks=[];ix=[];obs=eligible(p.data)
    for operation in ['delete','retain']:
        for f in GRID:
            for i,order in enumerate(orders):
                take=order[:int(np.ceil(f*len(order)))];m=np.ones((3,50),bool)
                if operation=='delete':m[:,take]=False
                else:
                    m[obs[i]]=False;m[:,take]=True
                masks.append(m);ix.append(i)
    pred=p.forward(ix,masks)
    result={k:v.reshape(2,len(GRID),len(orders),*v.shape[1:]) for k,v in pred.items()}
    np.savez_compressed(path,**result)
    print('fidelity',name,flush=True)
    return result


def confidence_interval(values, draws):
    return dict(mean=float(np.mean(values)),ci95=np.quantile(np.asarray(values)[draws].mean(1),[.025,.975]).tolist())


def fidelity(p,e):
    n=len(e['y']);classes=e['probs'].argmax(1);obs=e['eligible'].any(1)
    random=[]
    for rep in range(5):
        rng=np.random.default_rng(SEED+100+rep)
        orders=[rng.permutation(np.flatnonzero(row)) for row in obs]
        random.append(curve(p,orders,f'random_{rep}'))
    summary={};rows=[];bs=np.random.default_rng(SEED).integers(0,n,(500,n))
    for target in ['intensity','classification']:
        selected=curve(p,score_order(e,target),target)
        def delta(pred):
            if target=='intensity':return abs(pred['y']-e['y'])
            pc=np.take_along_axis(pred['probs'],classes[None,None,:,None],-1)[...,0]
            return e['probs'][np.arange(n),classes]-pc
        chosen=delta(selected);rand=np.stack([delta(r) for r in random])
        auc=np.trapezoid(chosen,GRID,axis=1)/GRID[-1]
        rand_auc=np.trapezoid(rand,GRID,axis=2).mean(0)/GRID[-1]
        summary[target]={}
        for op,name in enumerate(['delete','retain']):
            summary[target][name]=dict(selected=confidence_interval(auc[op],bs),
                random=confidence_interval(rand_auc[op],bs),paired_difference=confidence_interval(auc[op]-rand_auc[op],bs))
            for k,f in enumerate(GRID):
                rows.append(dict(target=target,operation=name,fraction=f,selected=float(chosen[op,k].mean()),
                                 random=float(rand[:,op,k].mean())))
    pd.DataFrame(rows).to_csv(OUT3/'validation/fidelity_curves.csv',index=False)
    write_json(OUT3/'validation/fidelity_summary.json',summary)


def decision(p,e,split):
    path=OUT3/f'{split}_decision.json'
    if path.exists():return json.loads(path.read_text())
    base=e['probs'].argmax(1);obs=e['eligible'].any(1);ks=[];paths=[]
    for rep in range(6):
        checkpoint=OUT3/'validation'/f'decision_{split}_{rep}.json'
        checkpoint.parent.mkdir(exist_ok=True)
        if checkpoint.exists():
            cached=json.loads(checkpoint.read_text());ks.append(cached['k'])
            if rep==0:paths=cached['paths']
            continue
        rng=np.random.default_rng(SEED+200+rep)
        orders=score_order(e,'classification') if rep==0 else [rng.permutation(np.flatnonzero(o)) for o in obs]
        masks=[];ix=[]
        for i,order in enumerate(orders):
            for k in range(1,len(order)+1):
                m=np.ones((3,50),bool);m[:,order[:k]]=False;masks.append(m);ix.append(i)
        pred=p.forward(ix,masks);start=0;kk=[];pp=[]
        for i,order in enumerate(orders):
            n=len(order);prob=pred['probs'][start:start+n];y=pred['y'][start:start+n];start+=n
            flips=np.flatnonzero(prob.argmax(-1)!=base[i]);kk.append(int(flips[0]+1) if len(flips) else n+1)
            if rep==0:pp.append(dict(positions=order.tolist(),probabilities=prob.tolist(),intensity=y.tolist()))
        ks.append(kk)
        if rep==0:paths=pp
        write_json(checkpoint,dict(k=kk,paths=pp))
        print('decision',split,rep,flush=True)
    length=obs.sum(1);kk=np.asarray(ks)
    result=dict(ids=e['ids'].tolist(),flip_k=ks[0],flip_censored=(kk[0]>length).tolist(),
        flip_k_rand=kk[1:].mean(0).tolist(),random_k=ks[1:],paths=paths,
        content_length=length.tolist(),target='final_fused_probability_argmax',
        definition='First class change along ranked joint-content deletion; no flip encoded length+1; special tokens retained.')
    write_json(path,result);return result


def summarize(p,e,split):
    q=e['probs'];phi=e['shapley_y'];pc=e['shapley_probs'];c=q.argmax(1)
    class_phi=pc[np.arange(len(c)),:,c]
    residual=float(abs(phi.sum(1)-(e['subset_y'][:,7]-e['subset_y'][:,0])).max())
    prob_residual=float(abs(pc.sum(1)-(e['subset_probs'][:,7]-e['subset_probs'][:,0])).max())
    additive=float(abs(e['scaled_core_contributions'].sum(1)+e['scaled_core_interaction']+e['expert_contribution']-e['y']).max())
    assert residual<1e-8 and prob_residual<1e-8 and additive<2e-5
    assert np.isfinite(q).all() and np.allclose(q.sum(1),1)
    summary=dict(n=len(q),shapley_intensity_efficiency_max=residual,
        shapley_probability_efficiency_max=prob_residual,additive_residual_max=additive,
        main_intensity_counts={m:int((abs(phi).argmax(1)==j).sum()) for j,m in enumerate(['text','audio','vision'])},
        main_classification_counts={m:int((abs(class_phi).argmax(1)==j).sum()) for j,m in enumerate(['text','audio','vision'])})
    if split=='valid':
        summary['metrics']=metrics(p.data['y'],q,e['y'])
        errors=c!=(np.sign(p.data['y']).astype(int)+1)
        margin=np.sort(q,axis=1)[:,-1]-np.sort(q,axis=1)[:,-2]
        frame=pd.DataFrame(dict(id=e['ids'],error=errors,mae=abs(e['y']-p.data['y']),margin=margin,
            main_modality=np.asarray(['text','audio','vision'])[abs(class_phi).argmax(1)],
            text_vision_conflict=class_phi[:,0]*class_phi[:,2]<0))
        rows=[]
        for key in ['main_modality','text_vision_conflict','margin']:
            group=pd.qcut(frame[key],4,duplicates='drop') if key=='margin' else frame[key]
            for name,sub in frame.groupby(group,observed=True):
                rows.append(dict(variable=key,stratum=str(name),n=len(sub),error_rate=float(sub.error.mean()),mae=float(sub.mae.mean())))
        pd.DataFrame(rows).to_csv(OUT3/'validation/error_strata.csv',index=False)
    write_json(OUT3/f'{split}_summary.json',summary)
    return summary


def main():
    global OUT3
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--stage',choices=['all','attachment4','valid','report'],default='all')
    ap.add_argument('--skip-pdf',action='store_true',help='Compatibility flag; the COLING manuscript is built separately')
    ap.add_argument('--skip-figures',action='store_true',help='Export numerical results without rendering supplementary figures')
    ap.add_argument('--output-root',type=Path,help='Fresh output directory; defaults to outputs/reproduced/explanations_<UTC>')
    ap.add_argument('--resume',action='store_true',help='Resume an explicitly selected reproduction directory')
    args=ap.parse_args()
    original = ROOT/'outputs/problem3'
    OUT3 = (args.output_root or ROOT/'outputs/reproduced'/('explanations_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    if OUT3 == original.resolve() or original.resolve() in OUT3.parents:
        ap.error('Use an independent output directory; bundled historical explanation records are preserved.')
    if OUT3.exists() and any(OUT3.iterdir()) and not (args.resume or args.stage == 'report'):
        ap.error('Output directory is not empty; choose another directory or specify --resume.')
    os.environ['COLING_P3_OUTPUT_ROOT'] = str(OUT3)
    import p3_common, p3_fusion
    p3_common.OUT3 = p3_fusion.OUT3 = OUT3
    torch.set_num_threads(args.threads);OUT3.mkdir(parents=True,exist_ok=True)
    if not (OUT3/'alignment').exists() and (original/'alignment').is_dir():
        shutil.copytree(original/'alignment',OUT3/'alignment')
    if not all((OUT3/'cache'/f'{s}.npz').exists() or (ROOT/'outputs/problem2/cache'/f'{s}.npz').exists()
               for s in ['valid','test','attachment4']):
        subprocess.run([sys.executable,str(ROOT/'src/p2p3/prepare_p3.py'),'--device',args.device],check=True)
    if not all((OUT3/'alignment'/f'{i:02d}.json').exists() for i in range(1,21)):
        subprocess.run([python_executable('alignment'),str(ROOT/'src/p2p3/align_attachment4.py')],check=True)
    fixed_protocol();started=time.time()
    if args.stage!='report':
        p=FusionExplainer(args.device)
        for split in ['attachment4','valid']:
            if args.stage not in ['all',split]:continue
            d=dataset(split);p.prepare(d,split)
            if split=='attachment4':validate_public(p,d)
            e=explain(p,split)
            if split=='valid':fidelity(p,e)
            decision(p,e,split);summarize(p,e,split)
            from export_p3_fusion import export
            export(d,e,split)
            write_json(OUT3/'runtime/progress.json',dict(last_completed_split=split,elapsed_seconds=time.time()-started))
    if args.stage in ['all','report']:
        from validate_p3_fusion import real_deletion
        real_deletion(args.device)
        write_json(OUT3/'report/numerical_summary.json', {split: json.loads((OUT3/f'{split}_summary.json').read_text()) for split in ['valid','attachment4']})
        if not args.skip_figures:
            for script in ['figures_p3_fusion.py', 'figures_p3_completion.py']:
                subprocess.run([python_executable('alignment'), str(ROOT/'src/p2p3'/script)], cwd=ROOT, check=True)
    print(f'Explanation outputs: {OUT3}',flush=True)


if __name__=='__main__':main()
