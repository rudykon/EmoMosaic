"""Small real-data integration check; no model selection or reported F1 score.

Uses 96 train examples for disposable heads, then checks nine raw-file/masking
cases against independently encoded features and cached RM core predictions.
"""
import argparse
import json
import time

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from project_paths import DATA_ROOT
from data import ROOT,load_split
from improvement_common import EXP,save_json,pool_tokens,masked,text_observed_fraction,fusion,digest
from improvement_evaluation import CachedEvaluator,base_config
from extract_text_experts import read_text
from improved_predict import ImprovedPredictor
from probe_text_experts import llm_observed
from qwen35_worker import Qwen35Worker
from text_encoder_registry import encoder_specs
from train import validation_masks


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--encoder',choices=['qwen35_08','qwen35_2','qwen35_4'],default='qwen35_08')
    ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args();torch.set_num_threads(args.threads);started=time.time()
    spec=encoder_specs()[args.encoder];layer=12
    out=EXP/'pipeline_smoke'/args.encoder;out.mkdir(parents=True,exist_ok=True)
    assert (EXP/f'{args.encoder}_model_verification.json').exists()
    bert=AutoTokenizer.from_pretrained(ROOT/'models/pretrained/bert',local_files_only=True)
    worker=Qwen35Worker(args.encoder,ROOT/spec['local_dir'],threads=args.threads)
    def features(split,count,masks):
        data,texts,enc=read_text(split,bert,'decoded_text_bert_all_splits')
        pooled=[[] for _ in masks]
        for start in range(0,count,16):
            stop=min(start+16,count);values=worker.encode(texts[start:stop],layer)
            for result,mask in zip(pooled,masks):
                observed=data['obs'][start:stop,0]&(data['token_ids'][start:stop]!=100)
                if mask is not None:observed&=mask[start:stop,0]
                keep=np.stack([llm_observed(off,original,obs) for off,original,obs in
                               zip(values['offsets'],enc['offset_mapping'][start:stop],observed)])
                keep&=values['attention_mask'].astype(bool)
                result.append(pool_tokens(values['hidden'],keep))
            print(f'{split}: {stop}/{count}',flush=True)
        return [np.concatenate(v) for v in pooled]
    tr=load_split('train');x=features('train',96,[None])[0]
    labels=np.sign(tr['y'][:96]).astype(int)+1;assert set(labels)=={0,1,2}
    scaler=StandardScaler().fit(x);x=scaler.transform(x)
    head={'scaler':scaler,'classifier':LogisticRegression(C=.001,class_weight='balanced',max_iter=3000).fit(x,labels),
          'ridge':Ridge(alpha=1000.).fit(x,tr['y'][:96]),
          'metadata':{'integration_only':True,'layer':layer,'fit_split':'train','fit_indices':list(range(96))}}
    joblib.dump(head,out/'disposable_head.joblib')
    config=base_config()
    config['expert']={'kind':'linear_text','encoder':args.encoder,'layer':layer,'pooling':'mean_max_min',
                      'head':str((out/'disposable_head.joblib').relative_to(ROOT)),'model_dir':spec['local_dir']}
    config['fusion']={'w0':.5,'k':2.,'bias':[0.,0.,0.]};config['regression_w0']=.3;config['regression_k']=2.
    save_json(out/'decision_rules.json',config)
    special=load_split('attachment3');absent=np.ones_like(special['obs']);absent[:,0]=False
    masks=[None,validation_masks(special['lengths'],20260923)[1],absent]
    xs=features('attachment3',3,masks);worker.close()
    predictor=ImprovedPredictor(out/'decision_rules.json');evaluator=CachedEvaluator(args.threads)
    paths=sorted((DATA_ROOT / 'source_data').glob('attachment3*/aligned/*.pkl'))[:3]
    rows=[]
    for name,mask,values in zip(['natural','mixed40','text_absent'],masks,xs):
        transformed=scaler.transform(values)
        ep=head['classifier'].predict_proba(transformed);er=np.clip(head['ridge'].predict(transformed),-3,3)
        core=evaluator.core(config,'attachment3',mask)
        fraction=text_observed_fraction(masked(special,mask))[:3]
        expected,_=fusion(ep,core['probabilities'][:3],fraction,.5,2.,[0.,0.,0.])
        reg_weight=.3*fraction**2
        intensity=reg_weight*er+(1-reg_weight)*core['intensity'][:3]
        actual=[predictor.predict(predictor.load_file(path),None if mask is None else mask[i:i+1])
                for i,path in enumerate(paths)]
        p=np.concatenate([r['probs'] for r in actual]);y=np.concatenate([r['y'] for r in actual])
        np.testing.assert_allclose(p,expected,atol=.002,rtol=0)
        np.testing.assert_allclose(y,intensity,atol=.002,rtol=0)
        np.testing.assert_array_equal(p.argmax(1),expected.argmax(1))
        if name=='text_absent':
            for r in actual:
                np.testing.assert_array_equal(r['expert_weight'],0.)
                np.testing.assert_allclose(r['y'],r['core_y'])
        rows.append({'condition':name,'samples':3,'class_disagreements':0,
                     'max_probability_difference':float(np.max(np.abs(p-expected))),
                     'max_intensity_difference':float(np.max(np.abs(y-intensity)))})
    predictor.expert_model.close()
    result={'passed':True,'encoder':args.encoder,'revision':spec['revision'],'train_samples':96,
            'fixed_layer':layer,'trainable_backbone_parameters':0,'checks':rows,'seconds':time.time()-started,
            'selection_used':False,'test_labels_read':False,'not_a_performance_estimate':True,
            'config_sha256':digest(out/'decision_rules.json')}
    save_json(out/'complete.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
