"""Frozen-only clean/ablation and missing-grid evaluation, with video clustering.

Refuses to read test before verifying frozen configuration and model hashes.
Reports known prior test exposure; no selection or parameter fitting occurs.
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from scipy.stats import binomtest

from data import ROOT,OUT,MODALITIES
from missing import TYPES,fixed_mask
from train import validation_masks
from compare_boosted import make_mask
from improvement_common import EXP,save_json,digest,metrics
from improvement_evaluation import CachedEvaluator
from analyze_boosted_comparison import grouped_confusion,paired_comparison,holm,cm_metrics


def paired_mean_views(a,b,weights,signs):
    """Average view metrics inside each paired video resample, never pool views."""
    point_a=[cm_metrics(x.sum(0)) for x in a];point_b=[cm_metrics(x.sum(0)) for x in b]
    boot_a=[cm_metrics(weights@x) for x in a];boot_b=[cm_metrics(weights@x) for x in b]
    result={}
    for metric in ['macro_f1','accuracy','weighted_f1']:
        av=np.mean([v[metric] for v in boot_a],axis=0);bv=np.mean([v[metric] for v in boot_b],axis=0)
        pa=float(np.mean([v[metric] for v in point_a]));pb=float(np.mean([v[metric] for v in point_b]))
        result[metric]={'new':pa,'baseline':pb,'delta':pa-pb,'delta_ci95':np.quantile(av-bv,[.025,.975]).tolist(),
                        'new_ci95':np.quantile(av,[.025,.975]).tolist(),'baseline_ci95':np.quantile(bv,[.025,.975]).tolist()}
    perm=[]
    for x,y in zip(a,b):
        center=(x.sum(0)+y.sum(0))/2;shift=signs@(x-y)/2
        perm.append(cm_metrics(center+shift)['macro_f1']-cm_metrics(center-shift)['macro_f1'])
    values=np.mean(perm,axis=0);delta=result['macro_f1']['delta']
    result['macro_f1']['cluster_permutation_p_two_sided']=float((np.count_nonzero(np.abs(values)>=abs(delta)-1e-14)+1)/(len(values)+1))
    return result


def verify_freeze():
    folder=EXP/'final_selection'
    frozen=json.loads((folder/'frozen.json').read_text())
    assert digest(folder/'decision_rules.json')==frozen['decision_sha256']
    assert digest(folder/'validation_candidates.json')==frozen['candidate_table_sha256']
    if digest(EXP/'protocol.json')!=frozen['protocol_sha256']:raise RuntimeError('Frozen protocol changed')
    protected={**frozen['evaluation_assets_sha256'],**frozen['required_evidence_sha256'],**frozen['inference_source_sha256'],
               **frozen['input_sha256'],**frozen['feature_metadata_sha256']}
    for path,expected in protected.items():
        if digest(ROOT/path)!=expected:raise RuntimeError(f'Frozen asset changed: {path}')
    rows=json.loads((folder/'validation_candidates.json').read_text())
    # Feature extraction touches no labels; all needed feature caches must exist
    # before this evaluation can load test labels.
    encoders={r['config']['expert']['encoder'] for r in rows if r['config'].get('expert',{} ) and r['config']['expert']['kind']=='linear_text'}
    for encoder in encoders:
        for split in ['valid','test']:
            base=EXP/'features'/encoder/split
            if not (base/'complete.json').exists():
                raise RuntimeError(f'Missing frozen {encoder}/{split} feature cache')
            manifest=json.loads((base/'complete.json').read_text())
            for key,path in [('hidden_sha256',base/'hidden.npy'),('alignment_sha256',base/'alignment.npz'),
                             ('encoder_sha256',base.parent/'encoder.json')]:
                if digest(path)!=manifest[key]:raise RuntimeError(f'Feature cache changed: {path}')
    return frozen,{r['name']:r for r in rows}


def scenarios(data,replays,attachment3):
    if len(replays)!=30 or len(attachment3['files'])!=30:raise ValueError('Expected exactly 30 attachment3 replay patterns')
    if [r['file'] for r in replays]!=attachment3['files'].tolist():raise ValueError('Replay and attachment3 token orders differ')
    if [r['valid_len'] for r in replays]!=attachment3['lengths'].tolist():raise ValueError('Replay reference lengths differ')
    yield 'clean','clean',None
    yield 'mixed40','mixed',validation_masks(data['lengths'],20260923)[1]
    for kind in TYPES:
        for rate in [.2,.4,.6]:
            yield f'{kind}_{rate:.1f}','rate',fixed_mask(data['lengths'],kind,rate,20260923)
    for index,record in enumerate(replays):
        mask=make_mask(data,{'family':'replay','pattern_index':index},replays)
        yield f'replay_zero_{index:02d}','replay_zero',mask
        # The audit found 131 censored [UNK] slots. Add a separately identified
        # approximation of those positions to the historical zero-pattern replay.
        extended=mask.copy()
        unknown=np.flatnonzero(attachment3['token_ids'][index]==100)
        for i,length in enumerate(data['lengths']):
            for pos in unknown:
                lo=int(np.floor(pos/record['valid_len']*length))
                hi=min(length,int(np.ceil((pos+1)/record['valid_len']*length)))
                extended[i,0,lo:hi]=False
        yield f'replay_zero_unk_{index:02d}','replay_zero_unk',extended
    text_absent=np.ones_like(data['obs']);text_absent[:,0]=False
    yield 'text_absent','boundary',text_absent
    yield 'all_absent','boundary',np.zeros_like(data['obs'])


def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--bootstrap',type=int,default=5000)
    args=ap.parse_args()
    frozen,candidates=verify_freeze()
    out=EXP/'final_evaluation';out.mkdir(parents=True,exist_ok=True)
    protocol={'freeze_sha256':digest(EXP/'final_selection/frozen.json'),'bootstrap_replicates':args.bootstrap,
              'bootstrap_seed':20260926,'resampling_unit':'original_video','mcnemar':'exact clip-level; descriptive because clips share videos',
              'test_exposure':'Historical test results known; this round never selects on test',
              'families':['clean','mixed','7_types_x_3_rates','30_zero_replays','30_zero_plus_UNK_replays','text_absent','all_absent']}
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:
        raise RuntimeError('Evaluation protocol changed; preserve old results and use a versioned amendment')
    save_json(out/'protocol.json',protocol)
    evaluator=CachedEvaluator(args.threads)
    final=frozen['selected']['name'];baseline='original_rm'
    names=list(dict.fromkeys([baseline,*frozen['ablation_names'],final]))
    predictions=out/'predictions';predictions.mkdir(exist_ok=True)
    grid=[];headline=[];statistics=[]
    replays=json.loads((OUT/'data_audit.json').read_text())['special']['attachment3']
    special=evaluator.dataset('attachment3')
    for split in ['valid','test']:
        data=evaluator.dataset(split)
        for scenario,family,mask in scenarios(data,replays,special):
            # Full ablations at both headline conditions; key grid for final and baseline.
            compared=names if scenario in ['clean','mixed40'] else list(dict.fromkeys([baseline,final]))
            for name in compared:
                row=candidates[name]
                path=predictions/f'{split}_{scenario}_{name}.npz'
                if path.exists():
                    with np.load(path) as z:
                        if not np.array_equal(z['ids'],data['ids']):raise ValueError('Prediction cache IDs differ')
                        p,r=z['probabilities'],z['intensity']
                else:
                    if row['output']=='expert':
                        p,r=evaluator.expert(row['config'],split,mask)
                    else:
                        result=evaluator.predict(row['config'],split,mask)
                        p,r=result['probabilities'],result['intensity']
                    np.savez_compressed(path,ids=data['ids'],probabilities=p,intensity=r)
                m=metrics(data['y'],p,r)
                if not np.isfinite(p).all() or not np.isfinite(r).all():raise ValueError('Nonfinite prediction')
                np.testing.assert_allclose(p.sum(1),1.,atol=1e-6,rtol=1e-6)
                flat={'split':split,'scenario':scenario,'family':family,'model':name,
                      **{k:m[k] for k in ['accuracy','macro_f1','mae','corr','corr_defined']}}
                grid.append(flat)
                if scenario in ['clean','mixed40']:headline.append(flat)
            save_json(out/'progress.json',{'last_split':split,'last_scenario':scenario,'metric_rows':len(grid)})
            print(f'{split}/{scenario}: {len(compared)} frozen candidates',flush=True)
        labels=np.sign(data['y']).astype(int)+1
        _,group_index=np.unique([str(i).split('$_$')[0] for i in data['ids']],return_inverse=True)
        n_groups=int(group_index.max()+1)
        rng=np.random.default_rng(20260926)
        weights=rng.multinomial(n_groups,np.full(n_groups,1/n_groups),size=args.bootstrap)
        signs=rng.choice([-1,1],size=(args.bootstrap,n_groups))
        headline_cms={name:[] for name in names}
        for scenario in ['clean','mixed40']:
            with np.load(predictions/f'{split}_{scenario}_{baseline}.npz') as z:
                base=z['probabilities'].argmax(1)
            base_cm=grouped_confusion(labels,base,group_index,n_groups)
            for name in names:
                with np.load(predictions/f'{split}_{scenario}_{name}.npz') as z:
                    predicted=z['probabilities'].argmax(1)
                cm=grouped_confusion(labels,predicted,group_index,n_groups)
                headline_cms[name].append(cm)
                stat=paired_comparison(cm,base_cm,weights,signs)
                n01=int(((base!=labels)&(predicted==labels)).sum())
                n10=int(((base==labels)&(predicted!=labels)).sum())
                mcnemar=float(binomtest(n01,n01+n10,.5).pvalue) if n01+n10 else 1.
                statistics.append({'split':split,'scenario':scenario,'model':name,'n_videos':n_groups,
                                   'metrics':stat,'mcnemar_exact_p':mcnemar,'baseline_wrong_new_right':n01,
                                   'baseline_right_new_wrong':n10})
        for name in names:
            statistics.append({'split':split,'scenario':'mean_clean_mixed40','model':name,'n_videos':n_groups,
                               'metrics':paired_mean_views(headline_cms[name],headline_cms[baseline],weights,signs),
                               'aggregation':'Arithmetic mean of two view metrics inside each paired video resample'})
    primary=[r for r in statistics if r['split']=='test' and r['model']==final and r['scenario'] in ['clean','mixed40']]
    adjusted=holm([r['metrics']['macro_f1']['cluster_permutation_p_two_sided'] for r in primary])
    for row,value in zip(primary,adjusted):row['test_two_headlines_holm_p']=float(value)
    csv_write(out/'all_metrics.csv',grid);csv_write(out/'headline_metrics.csv',headline)
    save_json(out/'statistics.json',statistics)
    save_json(out/'complete.json',{'frozen_sha256':digest(EXP/'final_selection/frozen.json'),
                                  'metric_rows':len(grid),'headline_rows':len(headline),
                                  'prediction_files':len(list(predictions.glob('*.npz'))),
                                  'historical_test_exposure':True,'selection_performed':False})


if __name__=='__main__':
    main()
