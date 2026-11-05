"""Freeze full exploration using validation only, after every training branch ends.

Checkpoint probabilities are recomputed from the actual saved weights, not the
pre-serialization training state. Test is not opened anywhere in this script.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json

import numpy as np

from data import ROOT
from text_encoder_registry import encoder_specs,active_encoders,probe_encoders
from train import validation_masks
from improvement_common import (EXP, initialize_protocol, save_json, digest, metrics,
                                score_views, apply_bias, select_bias, fusion, masked, text_observed_fraction)
from improvement_evaluation import CachedEvaluator,base_config


def readiness(protocol,strict=True):
    required=[EXP/'existing/complete.json',EXP/'lora/ensemble_validation.json',EXP/'lora/group_cv/complete.json',
              EXP/'stage4_decision.json',EXP/'bert_svd_protocol.json',EXP/'qwen35_protocol_amendment.json',EXP/'qwen35_2b_protocol_amendment.json',EXP/'pipeline_smoke/qwen35_08/complete.json']
    for weight in protocol['classification_weights']:
        required += [EXP/'runs'/f'ce{weight:g}_s{s}'/'complete.json' for s in protocol['training_seeds']]
    required += [EXP/'runs'/f'mix_ce0.3_s{s}'/'complete.json' for s in protocol['training_seeds'][:5]]
    required += [EXP/'lora'/f'seed{s}'/'complete.json' for s in protocol['training_seeds'][:5]]
    for encoder in probe_encoders():
        required += [EXP/'probes'/encoder/'complete.json',EXP/'probes'/encoder/'selection.json',
                     EXP/'probes'/encoder/'train_group_cv.json']
        selection=EXP/'probes'/encoder/'selection.json'
        if selection.exists():
            selected=json.loads(selection.read_text())
            if encoder.startswith('bert') or selected['fusion_gate_passed']:
                required.append(EXP/'probes'/encoder/'strict_valid.json')
            else:
                required.append(EXP/'preliminary_fusion'/encoder/'decision.json')
    for encoder in active_encoders():
        if encoder.startswith('qwen35_'):
            required += [EXP/f'{encoder}_download_verified.json',EXP/f'{encoder}_model_verification.json']
    missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()]
    save_json(EXP/'freeze_readiness.json',{'ready':not missing,'missing':missing,'required_files':len(required)})
    if missing and strict:raise RuntimeError(f'{len(missing)} prerequisites incomplete; see freeze_readiness.json')
    for encoder in probe_encoders():
        if not (EXP/'probes'/encoder/'complete.json').exists():continue
        complete=json.loads((EXP/'probes'/encoder/'complete.json').read_text())
        if not complete['full_prespecified_grid']:raise RuntimeError(f'{encoder} probe grid incomplete')
        source='bert' if encoder=='bert_svd' else encoder
        selected=json.loads((EXP/'probes'/encoder/'selection.json').read_text())
        if selected['encoder_manifest_sha256']!=digest(EXP/'features'/source/'encoder.json'):
            raise RuntimeError(f'{encoder} encoder changed after probe fitting')
    return [p for p in required if p.exists()]


def core_configs(protocol):
    configs={'original_rm':base_config()}
    for weight in protocol['classification_weights']:
        for count in [5,15]:
            c=base_config()
            c['core_checkpoints']=[f'outputs/problem2/improvement/runs/ce{weight:g}_s{s}/best.pt'
                                   for s in protocol['training_seeds'][:count]]
            configs[f'ce{weight:g}_{count}seed']=c
    c=base_config()
    c['core_checkpoints']=[f'outputs/problem2/improvement/runs/mix_ce0.3_s{s}/best.pt'
                           for s in protocol['training_seeds'][:5]]
    c['layer_norm_stats']='outputs/problem2/improvement/features/bert/norm_stats.json'
    configs['bert_scalar_mix_5seed']=c
    return configs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--preview-completed',action='store_true',help='Summarize completed branches; never freezes or evaluates test')
    args=ap.parse_args()
    protocol=initialize_protocol()
    required=readiness(protocol,strict=not args.preview_completed)
    if args.check_only:
        print('All freeze prerequisites are complete')
        return
    out=EXP/('completed_branch_preview' if args.preview_completed else 'final_selection');out.mkdir(parents=True,exist_ok=True)
    if (out/'frozen.json').exists():
        print('Already frozen; refusing to retune')
        return
    evaluator=CachedEvaluator(args.threads)
    valid=evaluator.dataset('valid');y=valid['y']
    masks=[None,validation_masks(valid['lengths'],20260923)[1]]
    fraction=[text_observed_fraction(masked(valid,m)) for m in masks]
    rows=[];core_predictions={};ablations=[]

    def record(name,config,p,r,eligible=True,output='fused'):
        ms=[metrics(y,v,w) for v,w in zip(p,r)]
        row={'name':name,'primary_score':score_views(y,p),'clean':ms[0],'mixed40':ms[1],
             'eligible_for_final':eligible,'output':output,'config':deepcopy(config)}
        rows.append(row)
        np.savez_compressed(out/f'{name}_valid.npz',ids=valid['ids'],probabilities=p,intensity=r)
        save_json(out/'validation_candidates.json',rows)
        print(f'{name}: {row["primary_score"]:.6f}',flush=True)
        return row

    configs=core_configs(protocol)
    core_rows=[]
    for name,config in configs.items():
        predictions=[evaluator.core(config,'valid',m) for m in masks]
        p=[r['probabilities'] for r in predictions];r=[v['intensity'] for v in predictions]
        core_predictions[name]=(p,r)
        raw=record(name,config,p,r)
        calibrated=deepcopy(config);offset=select_bias(y,p)
        calibrated['fusion']['bias']=offset['bias']
        chosen=record(name+'_bias',calibrated,[apply_bias(v,offset['bias']) for v in p],r)
        core_rows.append((chosen,name))
        if name in ['original_rm','bert_scalar_mix_5seed']:ablations.append(raw['name'])
        if name in ['original_rm','bert_scalar_mix_5seed'] or name.endswith('15seed'):ablations.append(chosen['name'])
    best_core=max(core_rows,key=lambda v:v[0]['primary_score'])[1]
    selected_cores=list(dict.fromkeys(['original_rm',best_core]))
    ablations.append(best_core+'_bias')

    # Preserve the stage-0 audio/vision removal ablation in the frozen test table.
    # These classifiers borrow the original core intensity, which is labelled
    # explicitly in their names/configurations and never presented as SVD regression.
    for kind in ['all','text']:
        config=base_config();config['expert']={'kind':'svd','head':f'outputs/problem2/improvement/existing/svd_{kind}.joblib'}
        p=[evaluator.expert(config,'valid',m)[0] for m in masks]
        offset=select_bias(y,p);config['fusion']={'w0':1.,'k':1.,'bias':offset['bias']}
        row=record(f'svd_{kind}_bias',config,[apply_bias(v,offset['bias']) for v in p],core_predictions['original_rm'][1],False)
        ablations.append(row['name'])

    experts={}
    for encoder in probe_encoders():
        if args.preview_completed and not (EXP/'probes'/encoder/'complete.json').exists():continue
        selection=json.loads((EXP/'probes'/encoder/'selection.json').read_text())
        spec=selection['selected']
        e={'kind':'linear_text','encoder':'bert' if encoder=='bert_svd' else encoder,'layer':spec['layer'],'pooling':spec['pooling'],
           'head':f'outputs/problem2/improvement/probes/{encoder}/{spec["name"]}.joblib'}
        if not encoder.startswith('bert'):e['model_dir']=encoder_specs()[encoder]['local_dir']
        config=base_config();config['expert']=e
        predictions=[evaluator.expert(config,'valid',m) for m in masks]
        p=[v[0] for v in predictions];r=[v[1] for v in predictions]
        row=record(encoder+'_probe',config,p,r,False,'expert')
        ablations.append(row['name'])
        if encoder.startswith('bert') or selection['fusion_gate_passed']:experts[encoder]=(e,p,r)
    # Last-layer baseline uses the exact same frozen-cache/probe protocol.
    bert_search=json.loads((EXP/'probes/bert/search.json').read_text())
    last=max([r for r in bert_search if r['layer']==12],key=lambda r:r['primary_score'])
    config=base_config();config['expert']={'kind':'linear_text','encoder':'bert','layer':12,'pooling':last['pooling'],
                                         'head':f'outputs/problem2/improvement/probes/bert/{last["name"]}.joblib'}
    prediction=[evaluator.expert(config,'valid',m) for m in masks]
    record('bert_last_probe',config,[v[0] for v in prediction],[v[1] for v in prediction],False,'expert')
    ablations.insert(1,'bert_last_probe')

    lora={'kind':'bert_lora','adapters':[f'outputs/problem2/improvement/lora/seed{s}/best.pt' for s in protocol['training_seeds'][:5]]}
    config=base_config();config['expert']=lora
    predictions=[evaluator.expert(config,'valid',m) for m in masks]
    lp,lr=[v[0] for v in predictions],[v[1] for v in predictions]
    record('lora_probe',config,lp,lr,False,'expert');ablations.append('lora_probe')
    experts['lora']=(lora,lp,lr)

    for core_name in selected_cores:
        cp,cr=core_predictions[core_name]
        # SVD mixed-family ensemble remains a fully measured alternative.
        svd={'kind':'svd','head':'outputs/problem2/improvement/existing/svd_all.joblib'}
        config=deepcopy(configs[core_name]);config['expert']=svd
        sp=[evaluator.expert(config,'valid',m)[0] for m in masks]
        trials=[]
        for w in protocol['fusion_weights']:
            p=[w*s+(1-w)*c for s,c in zip(sp,cp)]
            b=select_bias(y,p);trials.append(dict(b,w0=w,k=1.))
        chosen=max(trials,key=lambda r:r['score']);config['fusion']=chosen
        p=[apply_bias(chosen['w0']*s+(1-chosen['w0'])*c,chosen['bias']) for s,c in zip(sp,cp)]
        record(core_name+'_svd_bias',config,p,cr)
        for expert_name,(e,ep,er) in experts.items():
            trials=[];raw_trials=[]
            for k in protocol['fusion_exponents']:
                for w in protocol['fusion_weights']:
                    p=[fusion(a,b,f,w,k)[0] for a,b,f in zip(ep,cp,fraction)]
                    raw_trials.append({'score':score_views(y,p),'w0':w,'k':k,'bias':[0.,0.,0.]})
                    trials.append(dict(select_bias(y,p),w0=w,k=k))
            name=f'{core_name}_{expert_name}_fusion'
            for suffix,choice in [('',max(raw_trials,key=lambda r:r['score'])),('_bias',max(trials,key=lambda r:r['score']))]:
                config=deepcopy(configs[core_name]);config['expert']=e;config['fusion']=choice
                p=[fusion(a,b,f,choice['w0'],choice['k'],choice['bias'])[0] for a,b,f in zip(ep,cp,fraction)]
                regression=[]
                for w in protocol['fusion_weights']:
                    weights=[np.where(f>0,w*f**choice['k'],0.) for f in fraction]
                    r=[a*v+(1-a)*c for a,v,c in zip(weights,er,cr)]
                    regression.append((float(np.mean([np.abs(y-v).mean() for v in r])),w,r))
                _,rw,r=min(regression,key=lambda r:r[0]);config.update(regression_w0=rw,regression_k=choice['k'])
                record(name+suffix,config,p,r);ablations.append(name+suffix)
            save_json(out/f'{name}_search.json',{'raw':raw_trials,'biased':trials})
    best=max([r for r in rows if r['eligible_for_final']],key=lambda r:r['primary_score'])
    decision=deepcopy(best['config']);decision.update(scope='final_validation_frozen',selection_metric=protocol['primary_metric'],selected_name=best['name'])
    if args.preview_completed:
        decision['scope']='preliminary_completed_branches_only'
        save_json(out/'decision_rules.json',decision)
        save_json(out/'preview.json',{'selected':best,'frozen':False,'test_evaluated':False,
                                     'warning':'Incomplete branches excluded; this is not final selection'})
        return
    save_json(out/'decision_rules.json',decision)
    assets=set(decision['core_checkpoints'])
    if decision.get('expert'):
        e=decision['expert']
        assets.update(e['adapters'] if e['kind']=='bert_lora' else [e['head']])
    assets.update(['models/rmnet/norm_stats.json'])
    if 'layer_norm_stats' in decision:assets.add(decision['layer_norm_stats'])
    evaluation_assets=set(assets)
    for row in rows:
        c=row['config'];evaluation_assets.update(c['core_checkpoints'])
        if 'layer_norm_stats' in c:evaluation_assets.add(c['layer_norm_stats'])
        if c.get('expert'):
            e=c['expert'];evaluation_assets.update(e['adapters'] if e['kind']=='bert_lora' else [e['head']])
    sources=['data.py','model.py','predict.py','improved_predict.py','improvement_evaluation.py',
             'improvement_common.py','probe_text_experts.py','extract_text_experts.py','missing.py',
             'text_encoder_registry.py','qwen35_worker.py','train.py','bert_layer_data.py','lora_text_expert.py','boosted_classifier.py','metrics.py']
    save_json(out/'frozen.json',{'selected':best,'test_evaluated_before_freeze':False,
                               'historical_test_exposure':True,'protocol_sha256':digest(EXP/'protocol.json'),
                               'decision_sha256':digest(out/'decision_rules.json'),
                               'candidate_table_sha256':digest(out/'validation_candidates.json'),
                               'required_evidence_sha256':{str(p.relative_to(ROOT)):digest(p) for p in required},
                               'assets_sha256':{p:digest(ROOT/p) for p in sorted(assets)},
                               'evaluation_assets_sha256':{p:digest(ROOT/p) for p in sorted(evaluation_assets)},
                               'input_sha256':{f'outputs/problem2/cache/{s}.npz':digest(ROOT/f'outputs/problem2/cache/{s}.npz')
                                               for s in ['train','valid','test','attachment3','attachment4']},
                               'feature_metadata_sha256':{str(p.relative_to(ROOT)):digest(p)
                                     for encoder in active_encoders()
                                     for p in [EXP/'features'/encoder/'encoder.json',
                                               EXP/'features'/encoder/'train/complete.json',EXP/'features'/encoder/'valid/complete.json']},
                               'inference_source_sha256':{f'src/p2p3/{p}':digest(ROOT/'src/p2p3'/p) for p in sources},
                               'ablation_names':list(dict.fromkeys(ablations))})


if __name__=='__main__':
    main()
