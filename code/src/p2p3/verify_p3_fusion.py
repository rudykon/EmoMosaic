"""Cross-artifact invariants; no historical thresholds or ordinal probabilities."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from project_paths import resolve_asset, PAPER_ROOT, PROJECT_ROOT
from data import ROOT, write_json
from p3_common import OUT3, digest, attachment_paths
from model_paths import asset_matches


def main(require_pdf=False, output_path=None):
    protocol=json.loads((OUT3/'explanation_rules.json').read_text())
    for name,value in protocol['model_assets'].items():assert asset_matches(ROOT/name,value),name
    for name,value in protocol['inputs'].items():assert digest(ROOT/name)==value,name
    for source in attachment_paths():
        sid=source.stem
        # Official filenames may include a prefix; use the alignment's own file field.
        matches=[json.loads(p.read_text()) for p in (OUT3/'alignment').glob('[0-9][0-9].json')]
        alignment=next(a for a in matches if Path(a['file']).name==source.name)
        assert digest(source)==alignment['feature_sha256'],source
        assert digest(resolve_asset(alignment['video']))==alignment['video_sha256'],alignment['video']
    counts={};max_residual=0.;changed={}
    for split,count in [('valid',728),('attachment4',20)]:
        name='attachment4_predictions_and_explanations.csv' if split=='attachment4' else 'validation_predictions_and_explanations.csv'
        frame=pd.read_csv(OUT3/name,dtype={'id':str})
        assert len(frame)==count and frame.id.is_unique
        probabilities=frame[['p_neg','p_neu','p_pos']].to_numpy()
        assert np.allclose(probabilities.sum(1),1) and (probabilities>=0).all()
        assert np.array_equal(np.asarray(['Negative','Neutral','Positive'])[probabilities.argmax(1)],frame.pred_polarity)
        paths=sorted((OUT3/'samples'/split).glob('*.json'));assert len(paths)==count
        for path in paths:
            s=json.loads(path.read_text());p=s['prediction'];assert s['schema']=='p3_fusion_evidence_v2'
            row=frame[frame.id==p['id']].iloc[0]
            assert abs(row.pred_intensity-p['pred_intensity'])<1e-12
            assert row.pred_polarity==p['pred_polarity']
            assert len(s['positions'])==3 and all(len(v)==50 for v in s['positions'])
            y=np.asarray(s['subset_intensity']);prob=np.asarray(s['subset_probabilities'])
            assert prob.shape==(8,3) and np.allclose(prob.sum(1),1)
            assert abs(y[7]-p['pred_intensity'])<2e-5
            assert np.max(abs(prob[7]-np.array([p['p_neg'],p['p_neu'],p['p_pos']])))<2e-5
            residual=max(abs(np.sum(s['shapley_intensity'])-(y[7]-y[0])),
                np.max(abs(np.sum(s['shapley_probabilities'],axis=0)-(prob[7]-prob[0]))))
            max_residual=max(max_residual,float(residual));assert residual<1e-10
            a=s['additive_intensity'];assert abs(sum(a['scaled_core_contributions'])+a['scaled_core_interaction']+a['expert_contribution']-p['pred_intensity'])<2e-5
            for modality in s['positions']:
                for r in modality:
                    mp=np.asarray(r['masked_probabilities'])
                    assert np.isclose(mp.sum(),1) and np.min(mp)>-1e-12
                    assert r['polarity_flip']==bool(mp.argmax()!=s['fixed_target_class'])
                    if not r['observed']:
                        assert r['intensity_delta']==0 and np.max(np.abs(r['probability_delta']))==0
            path_probs=np.asarray(s['flip_path']['probabilities']);n=len(path_probs)
            flips=np.flatnonzero(path_probs.argmax(1)!=s['fixed_target_class'])
            assert p['flip_k']==(int(flips[0]+1) if len(flips) else n+1)
            assert p['flip_censored']==(len(flips)==0)
        counts[split]=count
        if split=='attachment4':
            old=pd.read_csv(ROOT/'outputs/problem3_legacy_rm_20260927'/name,dtype={'id':str}) if (ROOT/'outputs/problem3_legacy_rm_20260927'/name).exists() else None
            if old is not None:
                assert old.id.tolist()==frame.id.tolist()
                changed=dict(class_changes=int((old.pred_polarity!=frame.pred_polarity).sum()),
                    mean_absolute_intensity_change=float(abs(old.pred_intensity-frame.pred_intensity).mean()))
    # Match the independently frozen P2 headline; only floating-point tolerance.
    headline=pd.read_csv(ROOT/'outputs/problem2/improvement/final_evaluation/headline_metrics.csv')
    expected=headline[(headline.model=='original_rm_lora_fusion_bias')&(headline.split=='valid')&(headline.scenario=='clean')].iloc[0]
    summary=json.loads((OUT3/'valid_summary.json').read_text())['metrics']
    differences={k:abs(float(expected[k])-summary[k]) for k in ['accuracy','macro_f1','mae','corr']}
    assert max(differences.values())<2e-5,differences
    assert json.loads((OUT3/'interface_verification.json').read_text())['passed']
    assert json.loads((OUT3/'validation/real_word_deletion.json').read_text())['n_pairs']==728
    subprocess.run([sys.executable,str(ROOT/'src/p2p3/test_p3_fusion.py')],check=True)
    result=dict(passed=True,schema='p3_final_fusion_v1',counts=counts,shapley_efficiency_max=max_residual,
        p2_frozen_headline_differences=differences,attachment4_changes_from_legacy=changed,
        model_hashes_unchanged=True,checks=['all 748 sample schemas, probabilities, all positions, subset efficiency, fusion additive reconstruction, final-probability flips',
            '12 independent raw-input public interface comparisons','frozen P2 validation headline equality','analytic Shapley games'],
        limitation='Independent full-model cross-seed/randomization checks remain unperformed; no raw human emotion causality claims.')
    if require_pdf:
        pdf=PAPER_ROOT/'main.pdf';assert pdf.is_file()
        result['paper_sha256']=digest(pdf)
    target = Path(output_path) if output_path else (PROJECT_ROOT/'validation/p3_reverification.json' if OUT3 == ROOT/'outputs/problem3' else OUT3/'verification.json')
    write_json(target,result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-pdf', action='store_true')
    parser.add_argument('--output', type=Path, help='Write checks separately from the preserved historical record')
    arguments=parser.parse_args()
    main(require_pdf=arguments.require_pdf, output_path=arguments.output)
