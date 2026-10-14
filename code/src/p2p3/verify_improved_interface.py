"""Compare independent raw-file and frozen-cache inference, including masks."""
import argparse
import json

import numpy as np
import torch

from project_paths import DATA_ROOT
from data import ROOT,load_split
from improvement_common import EXP,save_json,apply_bias,digest
from improved_predict import ImprovedPredictor
from improvement_evaluation import CachedEvaluator
from train import validation_masks


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    torch.set_num_threads(2)
    predictor=ImprovedPredictor(args.config)
    evaluator=CachedEvaluator(2)
    data=load_split('attachment3')
    paths=sorted((DATA_ROOT / 'source_data').glob('attachment3*/aligned/*.pkl'))
    cached_names = [str(name).replace('附件3_', 'attachment3_') for name in data['files']]
    assert [p.name for p in paths] == cached_names
    absent=np.ones_like(data['obs']);absent[:,0]=False
    masks=[None,validation_masks(data['lengths'],20260923)[1],absent]
    rows=[]
    for name,mask in zip(['natural','mixed40','text_absent'],masks):
        expected=evaluator.predict(predictor.config,'attachment3',mask)
        actual=[predictor.predict(predictor.load_file(path),mask[i:i+1] if mask is not None else None)
                for i,path in enumerate(paths)]
        p=np.concatenate([a['probs'] for a in actual]);y=np.concatenate([a['y'] for a in actual])
        pd=float(np.max(np.abs(p-expected['probabilities'])));yd=float(np.max(np.abs(y-expected['intensity'])))
        assert pd<.002 and yd<.002,(name,pd,yd)
        assert np.array_equal(p.argmax(1),expected['probabilities'].argmax(1)),name
        np.testing.assert_allclose(p.sum(1),1.,atol=1e-6)
        for a in actual:
            np.testing.assert_allclose(a['core_contributions'].sum(1)+a['core_interaction'],a['core_y'],atol=1e-6)
            if name=='text_absent' and (predictor.config.get('expert') or {}).get('kind')!='svd':
                np.testing.assert_array_equal(a['expert_weight'],0.)
                np.testing.assert_allclose(a['probs'],apply_bias(a['auxiliary_probs'],predictor.config['fusion']['bias']),rtol=1e-6,atol=1e-7)
                np.testing.assert_allclose(a['y'],a['core_y'])
        rows.append({'condition':name,'n':len(paths),'classification_disagreements':0,
                     'max_probability_difference':pd,'max_intensity_difference':yd})
        print(rows[-1],flush=True)
    save_json(args.output,{'config':args.config,'config_sha256':digest(args.config),'checks':rows,'probability_simplex':True,
                           'input_files_sha256':{p.name:digest(p) for p in paths},
                           'core_attribution_identity':True,'missing_text_fallback':True,
                           'method':'Raw reencoding and file preprocessing compared against separately pooled frozen caches and batched core inference'})


if __name__=='__main__':
    main()
