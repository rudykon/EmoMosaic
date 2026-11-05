"""Export final probabilities, two attribution targets and raw-media mappings."""
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from data import ROOT, MODALITIES, write_json
from p3_common import OUT3, ordering, token_mapping
from p3_fusion import LABELS
from export_attachment4 import merge_segments


def export(d,e,split):
    dec=json.loads((OUT3/f'{split}_decision.json').read_text())
    assert dec['ids']==d['ids'].tolist()
    tok=AutoTokenizer.from_pretrained(ROOT/'models/pretrained/bert',local_files_only=True)
    folder=OUT3/'samples'/split;folder.mkdir(parents=True,exist_ok=True);rows=[]
    for n,sid in enumerate(d['ids']):
        alignment=json.loads((OUT3/'alignment'/f'{sid}.json').read_text()) if split=='attachment4' else None
        mapping=alignment['positions'] if alignment else token_mapping(str(d['raw_text'][n]),d['token_ids'][n],tok)
        c=int(e['probs'][n].argmax());py=e['shapley_y'][n];pc=e['shapley_probs'][n,:,c]
        sy=abs(py)/max(abs(py).sum(),1e-12);sc=abs(pc)/max(abs(pc).sum(),1e-12)
        row=dict(id=str(sid),raw_text=str(d['raw_text'][n]),valid_len=int(d['lengths'][n]),
            model='RM+BERT_LoRA',pred_polarity=LABELS[c],pred_intensity=float(e['y'][n]),
            p_neg=float(e['probs'][n,0]),p_neu=float(e['probs'][n,1]),p_pos=float(e['probs'][n,2]),
            core_intensity=float(e['core_y'][n]),expert_intensity=float(e['expert_y'][n]),
            classification_text_weight=float(e['classification_weight'][n]),regression_text_weight=float(e['regression_weight'][n]),
            probability_margin=float(np.sort(e['probs'][n])[-1]-np.sort(e['probs'][n])[-2]),
            main_modality_intensity=MODALITIES[int(abs(py).argmax())] if abs(py).sum()>1e-12 else 'undetermined',
            main_modality_classification=MODALITIES[int(abs(pc).argmax())] if abs(pc).sum()>1e-12 else 'undetermined',
            classification_tie_flag=bool(np.sort(sc)[-1]-np.sort(sc)[-2]<.05),
            intensity_tie_flag=bool(np.sort(sy)[-1]-np.sort(sy)[-2]<.05),
            flip_k=dec['flip_k'][n],flip_censored=dec['flip_censored'][n],flip_k_rand=dec['flip_k_rand'][n],
            WER=alignment['wer_ctc'] if alignment else None,CTC_mean=alignment['align_conf'] if alignment else None,
            face_rate=alignment['face_rate'] if alignment else None,
            time_quality_flags=';'.join(alignment['flags']) if alignment else 'no_raw_validation_media',
            explanation_target='fused_intensity_and_fixed_original_class_probability',
            intervention='post_encoding_mask_with_recomputed_fusion_weights')
        positions=[];top={};top_y={}
        for j,m in enumerate(MODALITIES):
            for name,value in [('shap_intensity',py[j]),('share_intensity',sy[j]),('shap_class',pc[j]),('share_class',sc[j]),
                               ('core_gate',e['core_gate'][n,j]),('scaled_core_contribution',e['scaled_core_contributions'][n,j])]:
                row[f'{name}_{"TAV"[j]}']=float(value)
            pp=[]
            for pos in mapping:
                i=pos['position'];dp=e['occlusion_probs'][n,j,i];mp=e['probs'][n]-dp
                pp.append(dict(**pos,observed=bool(d['obs'][n,j,i]),eligible=bool(e['eligible'][n,j,i]),
                    intensity_delta=float(e['occlusion_y'][n,j,i]),class_probability_delta=float(dp[c]),
                    probability_delta=dp.tolist(),masked_probabilities=mp.tolist(),
                    polarity_flip=bool(mp.argmax()!=c)))
            order=ordering(e['occlusion_probs'][n,j,:,c],e['eligible'][n,j])
            oy=ordering(abs(e['occlusion_y'][n,j]),e['eligible'][n,j])
            top[m]=[pp[int(i)] for i in order[:3]];top_y[m]=[pp[int(i)] for i in oy[:3]];positions.append(pp)
            row[f'{m}_evidence']=json.dumps(top[m],ensure_ascii=False,separators=(',',':'))
        result=dict(schema='p3_fusion_evidence_v2',prediction={k:v for k,v in row.items() if not k.endswith('_evidence')},
            fixed_target_class=c,subset_intensity=e['subset_y'][n].tolist(),subset_probabilities=e['subset_probs'][n].tolist(),
            shapley_intensity=py.tolist(),shapley_probabilities=e['shapley_probs'][n].tolist(),
            positions=positions,top_evidence=top,top_intensity_evidence=top_y,
            joint_intensity_delta=e['occlusion_y'][n,3].tolist(),joint_probability_delta=e['occlusion_probs'][n,3].tolist(),
            additive_intensity=dict(scaled_core_contributions=e['scaled_core_contributions'][n].tolist(),
                scaled_core_interaction=float(e['scaled_core_interaction'][n]),expert_contribution=float(e['expert_contribution'][n])),
            flip_path=dec['paths'][n],random_flip_k=[r[n] for r in dec['random_k']],
            audio_segments=merge_segments(top['audio']),vision_segments=merge_segments(top['vision']),
            video=alignment['video'] if alignment else None,
            alignment_notice=alignment['limitation'] if alignment else 'No raw validation media; token positions only.',
            limitation='Contextual states retain original text context. Feature masking is not literal word deletion. Core gate is not final-model attribution. Probabilities are not independently calibrated.')
        if split=='valid':result['truth']=float(d['y'][n])
        write_json(folder/f'{sid}.json',result);rows.append(row)
    name='attachment4_predictions_and_explanations.csv' if split=='attachment4' else 'validation_predictions_and_explanations.csv'
    pd.DataFrame(rows).to_csv(OUT3/name,index=False,encoding='utf-8-sig')
    write_json(OUT3/f'{split}_export_manifest.json',dict(n=len(rows),ids=d['ids'].tolist(),csv=name,
        schema='p3_fusion_evidence_v2',protocol='explanation_rules.json'))
    print('exported final fusion',split,len(rows),flush=True)
