"""Export every prediction and position, with auditable raw-media evidence."""
import argparse
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from data import ROOT, MODALITIES, write_json
from metrics import classes
from fidelity_eval import load_e
from p3_common import OUT3, LABELS, dataset, eligible, scores, fusion, ordering, token_mapping, provenance


def merge_segments(evidence):
    """Merge consecutive selected original words, retain low-trust qualifiers."""
    unique={}
    for r in evidence:
        if r.get('time_start') is not None:unique.setdefault(r['word_index'],[]).append(r)
    groups=[]
    for wi in sorted(unique):
        r=unique[wi][0];positions=[v['position'] for v in unique[wi]]
        if groups and wi<=groups[-1]['last_word_index']+1:
            g=groups[-1];g['end']=max(g['end'],r['time_end']);g['positions'].extend(positions)
            g['last_word_index']=wi;g['timing_usable'] &= r['timing_usable']
        else:groups.append(dict(start=r['time_start'],end=r['time_end'],positions=positions,
                                last_word_index=wi,timing_usable=r['timing_usable']))
    return groups


def export(split):
    rule=json.loads((OUT3/'explanation_rules.json').read_text())
    rules=json.loads((ROOT/'models/rmnet/decision_rules.json').read_text())
    e,seeds=load_e(split,rule['width']);d=dataset(split);obs=eligible(d)
    ranked=scores(e,obs,rule['method']);fused=fusion(e['occlusion'],e['gradient'],e['alpha'],obs)
    decision=json.loads((OUT3/f'{split}_decision.json').read_text())
    assert decision['ids']==d['ids'].tolist()
    tok=AutoTokenizer.from_pretrained(ROOT/'models/pretrained/bert',local_files_only=True)
    out=OUT3/'samples'/split;out.mkdir(parents=True,exist_ok=True);rows=[]
    thresholds=rules['thresholds'];labels=classes(e['y'],thresholds)
    for n,sid in enumerate(d['ids']):
        alignment=None
        if split=='attachment4':
            alignment=json.loads((OUT3/'alignment'/f'{sid}.json').read_text());mapping=alignment['positions']
        else:mapping=token_mapping(str(d['raw_text'][n]),d['token_ids'][n],tok)
        y=float(e['y'][n]);phi=e['shapley'][n];total=float(abs(phi).sum())
        share=abs(phi)/total if total>1e-12 else np.zeros(3)
        main=int(abs(phi).argmax());internal=int(abs(e['contributions'][n]).argmax())
        temp=rules['ordinal_temperature'];pneg=1/(1+np.exp((y+thresholds[0])/temp));ppos=1/(1+np.exp((thresholds[1]-y)/temp))
        row=dict(id=str(sid),valid_len=int(d['lengths'][n]),word_count=len(str(d['raw_text'][n]).split()),
            raw_text=str(d['raw_text'][n]),pred_intensity=y,pred_polarity=LABELS[labels[n]],
            seed_sd=float(seeds['y'][:,n].std()),margin=float(min(abs(y+thresholds[0]),abs(y-thresholds[1]))),
            interaction=float(e['y_int'][n]),empty_value=float(e['values'][n,0]),
            main_modality_shap=MODALITIES[main] if total>1e-12 else 'undetermined',
            main_modality_contrib=MODALITIES[internal],agree_flag=bool(main==internal),
            tie_flag=bool(np.sort(share)[-1]-np.sort(share)[-2]<.05),
            flip_k=int(decision['flip_k'][n]),flip_censored=decision['flip_censored'][n],
            flip_k_rand=float(decision['flip_k_rand'][n]),flip_ratio=float(decision['flip_ratio'][n]),
            evidence_method=rule['method'],window_width=rule['width'],selected_k=rule['top_k'],
            WER=alignment['wer_ctc'] if alignment else None,CTC_mean=alignment['align_conf'] if alignment else None,
            face_rate=alignment['face_rate'] if alignment else None,
            time_quality_flags=';'.join(alignment['flags']) if alignment else 'validation_has_no_raw_media')
        for j,c in enumerate(['neg','neu','pos']):
            row['aux_p_'+c]=float(e['auxiliary_probs'][n,j]);row['ordinal_p_'+c]=float([pneg,1-pneg-ppos,ppos][j])
        all_positions=[];top={};modalities=[]
        for j,m in enumerate(MODALITIES):
            direction=int(np.sign(phi[j])*np.sign(y-e['values'][n,0]))
            r=dict(modality=m,gate=float(e['w'][n,j]),contribution=float(e['contributions'][n,j]),
                shapley=float(phi[j]),share=float(share[j]),direction=direction,
                sufficiency_gap=float(abs(e['values'][n,1<<j]-y)),
                necessity_gap=float(abs(y-e['values'][n,7^(1<<j)])))
            modalities.append(r)
            for name,key in [('gate','gate'),('contrib','contribution'),('shap','shapley'),('share','share'),('direction','direction'),('sufficiency','sufficiency_gap'),('necessity','necessity_gap')]:row[f'{name}_{"TAV"[j]}']=r[key]
            pp=[]
            for pos in mapping:
                i=pos['position'];delta=float(e['occlusion'][n,j,i])
                pp.append(dict(**pos,observed=bool(d['obs'][n,j,i]),eligible=bool(obs[n,j,i]),
                    occlusion_delta=delta,masked_prediction=y-delta,
                    polarity_flip=bool(classes(np.array([y-delta]),thresholds)[0]!=labels[n]),
                    gradient_input=float(e['gradient'][n,j,i]),attention=float(e['alpha'][n,j,i]),
                    rank_fusion=float(fused[n,j,i]),score=float(ranked[n,j,i]),
                    no_evidence_note='' if obs[n,j,i] else ('special_or_padding' if pos['special'] else f'no_{m}_evidence')))
            order=ordering(ranked[n,j],obs[n,j]);top[m]=[pp[int(i)] for i in order[:3]]
            denom=float(abs(e['occlusion'][n,j,obs[n,j]]).sum())
            r['top_k_occlusion_mass_share']=float(abs(e['occlusion'][n,j,order[:rule['top_k']]]).sum()/denom) if denom else 0.
            r['top3_occlusion_mass_share']=float(abs(e['occlusion'][n,j,order[:3]]).sum()/denom) if denom else 0.
            r['mass_share_note']='Sum of absolute overlapping-window effects; not an additive causal share.'
            all_positions.append(pp)
            row[m+'_evidence']=json.dumps(top[m],ensure_ascii=False,separators=(',',':'))
        time_order=ordering(abs(e['joint_occlusion'][n]),obs[n].any(0))
        time_evidence=[dict(**mapping[int(i)],joint_occlusion_delta=float(e['joint_occlusion'][n,i])) for i in time_order[:3]]
        row['time_evidence']=json.dumps(time_evidence,ensure_ascii=False,separators=(',',':'))
        row['notes']='Model dependence on frozen features; independent estimated word times; gate is not importance. Flip count is ranking-path minimum; no-flip coded length+1.'
        if alignment and alignment['flags']:row['notes']+=' '+row['time_quality_flags']
        result=dict(schema='p3_evidence_v1',prediction={k:v for k,v in row.items() if not k.endswith('_evidence')},
            modalities=modalities,subset_values={str(i):float(e['values'][n,i]) for i in range(8)},
            seed_predictions=seeds['y'][:,n].tolist(),top_evidence=top,positions=all_positions,
            time_evidence=time_evidence,joint_occlusion=e['joint_occlusion'][n].tolist(),
            audio_segments=merge_segments(top['audio']),vision_segments=merge_segments(top['vision']),
            flip_prediction_path=decision['prediction_paths'][n],
            random_flip_k=[r[n] for r in decision['random_k']],
            special_token_effects=[dict(position=i,delta=float(e['occlusion'][n,0,i]),width=rule['width']) for i in [0,int(d['lengths'][n]-1)]],
            video=alignment['video'] if alignment else None,
            alignment_notice=alignment['limitation'] if alignment else 'No raw validation video/timing available; relative positions only.')
        if split=='valid':result['truth']=float(d['y'][n])
        write_json(out/f'{sid}.json',result);rows.append(row)
    name='attachment4_predictions_and_explanations.csv' if split=='attachment4' else 'validation_predictions_and_explanations.csv'
    pd.DataFrame(rows).to_csv(OUT3/name,index=False,encoding='utf-8-sig')
    write_json(OUT3/f'{split}_export_manifest.json',dict(n=len(rows),ids=d['ids'].tolist(),
        csv=name,model_assets=provenance(),rules=rule))
    print('Exported',split,len(rows),flush=True)


def main():
    from p3_common import require_legacy_protocol
    require_legacy_protocol()
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['valid','attachment4'],default='attachment4')
    export(ap.parse_args().split)


if __name__=='__main__':main()
