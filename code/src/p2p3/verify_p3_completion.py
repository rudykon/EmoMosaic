"""Verify supplementary figures/tables against saved explanation evidence.

The COLING manuscript is checked separately; no competition template is needed.
"""
import json
import numpy as np
import pandas as pd
from data import write_json
from project_paths import resolve_asset
from p3_common import OUT3, digest


def main():
    assets=json.loads((OUT3/'paper_checks/paper_completion_assets.json').read_text())
    sample=json.loads((OUT3/'samples/attachment4/19.json').read_text())
    assert assets['source_sample_sha256']==digest(OUT3/'samples/attachment4/19.json')
    eligible=[i for i in range(50) if any(m[i]['eligible'] for m in sample['positions'])]
    assert len(eligible)==39
    for plot in assets['distributions']:
        assert plot['positions']==eligible
        key='class_probability_delta' if plot['target']=='classification' else 'intensity_delta'
        expected=[[m[i][key] if m[i]['eligible'] else np.nan for i in eligible] for m in sample['positions']]
        np.testing.assert_allclose(plot['values'],expected,rtol=0,atol=0,equal_nan=True)
        np.testing.assert_allclose(plot['per_modality_max_abs'],np.nanmax(abs(np.asarray(expected)),axis=1))
        assert (OUT3/'figures'/(plot['figure']+'.pdf')).is_file()
    assert assets['source_replay']['source_sha256']==digest(resolve_asset(sample['video']))
    assert all(r['timing_usable'] for m in ['text','audio','vision'] for r in sample['top_evidence'][m])
    frame=pd.read_csv(OUT3/'attachment4_evidence_summary.csv',dtype={'id':str},keep_default_na=False)
    assert frame.id.tolist()==[f'{i:02}' for i in range(1,21)]
    for row in frame.to_dict('records'):
        s=json.loads((OUT3/'samples/attachment4'/(row['id']+'.json')).read_text())
        for m in ['text','audio','vision']:
            items=s['top_evidence'][m]
            if not items:
                assert row[m+'_status']=='no_observed_evidence'
                continue
            r=items[0]
            assert row[m+'_word']==r['word'] and int(row[m+'_position'])==r['position']
            assert row[m+'_timing_usable']==r['timing_usable']
            for a,b in [('time_start','time_start'),('time_end','time_end'),('probability_delta','class_probability_delta')]:
                assert abs(float(row[m+'_'+a])-r[b])<1e-12
    assert frame.loc[frame.id=='13','vision_status'].item()=='no_observed_evidence'
    for sid in ['15','16']:
        row=frame[frame.id==sid].iloc[0]
        assert not row.audio_timing_usable and not row.vision_timing_usable
    error_ids=['f_ZJ7L14oYQ$_$21','EQxkXiWAm08$_$8','x2lOwQaAn4g$_$4']
    errors=[json.loads((OUT3/'samples/valid'/(sid+'.json')).read_text()) for sid in error_ids]
    np.testing.assert_allclose([s['truth'] for s in errors],[2.,1/3,0.],atol=1e-7)
    assert errors[2]['top_evidence']['text'][0]['class_probability_delta']<0
    cards=json.loads((OUT3/'cards/selection.json').read_text())['samples']
    layered=json.loads((OUT3/'paper_checks/layered_cards_verification.json').read_text())
    assert layered['passed'] and layered['cards']==len(cards)
    for card in layered['cards_checked']:
        assert digest(OUT3/'figures'/(card['figure']+'.pdf'))==card['pdf_sha256']
    result=dict(passed=True, scope='supplementary_explanation_assets',
                attachment4_rows=20, evidence_summary_rows=20,
                plotted_content_positions=len(eligible), distribution_targets=2,
                source_replay_sample='19', error_cases=3, low_trust_flags_preserved=True,
                cards_verified=len(cards), manuscript_modified=False,
                generated_assets_sha256=digest(OUT3/'paper_checks/paper_completion_assets.json'))
    write_json(OUT3/'paper_checks/supplement_completion_verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
