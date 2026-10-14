"""Fit train-only linear probes on frozen hidden states; rank using valid only."""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from data import load_split
from text_encoder_registry import encoder_specs
from improvement_common import (EXP, initialize_protocol, save_json, pool_tokens,
                                views, metrics, score_views, digest)


def llm_observed(offsets, bert_offsets, bert_observed):
    """Zero any LLM token overlapping an occluded BERT character span."""
    offsets, bert_offsets = np.asarray(offsets), np.asarray(bert_offsets)
    keep = offsets[:,1] > offsets[:,0]
    if not np.any(bert_observed):
        return np.zeros(len(offsets), bool)
    missing = bert_offsets[(~np.asarray(bert_observed)) & (bert_offsets[:,1]>bert_offsets[:,0])]
    if len(missing):
        overlap = (offsets[:,0,None] < missing[None,:,1]) & (offsets[:,1,None] > missing[None,:,0])
        keep &= ~overlap.any(1)
    return keep


def pooled(encoder, split, layer, obs, mode='mean_max_min'):
    if encoder=='bert_svd':encoder='bert'
    root = EXP/'features'/encoder
    if not (root/split/'complete.json').exists():
        raise RuntimeError(f'Incomplete feature extraction: {encoder}/{split}')
    identity = json.loads((root/'encoder.json').read_text())
    j = identity['layers'].index(layer)
    hidden = np.load(root/split/'hidden.npy', mmap_mode='r')
    result = []
    if encoder == 'bert':
        for start in range(0,len(obs),64):
            result.append(pool_tokens(hidden[start:start+64,j],obs[start:start+64],mode))
    else:
        with np.load(root/split/'alignment.npz') as z:
            ptr, offsets, bert_offsets = z['token_ptr'], z['offsets'], z['bert_offsets']
            ids=z['bert_token_ids']
        obs=obs & (ids!=100)
        for i in range(len(obs)):
            lo,hi=ptr[i:i+2]
            keep=llm_observed(offsets[lo:hi],bert_offsets[i],obs[i])
            result.append(pool_tokens(hidden[lo:hi,j][None],keep[None],mode))
    return np.concatenate(result)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--encoder',choices=['bert','bert_svd','qwen','qwen7','qwen35_08','qwen35_2','qwen35_4'],required=True)
    ap.add_argument('--layers',default=None,help='Default: every preregistered layer')
    args=ap.parse_args()
    protocol=initialize_protocol()
    out=EXP/'probes'/args.encoder
    out.mkdir(parents=True,exist_ok=True)
    layers=encoder_specs()['bert' if args.encoder=='bert_svd' else args.encoder]['layers']
    if args.encoder=='bert_svd':
        middle=json.loads((EXP/'probes/bert/selection.json').read_text())['selected']['layer']
        layers=list(dict.fromkeys([middle,12]))
        amendment={'route':'BERT middle/top MMM followed by train-only SVD and balanced linear heads',
                   'layers':layers,'layer_rule':'validation-selected BERT layer and final layer',
                   'pooling':'mean_max_min','svd_components':384,'svd_seed':7,'svd_iterations':5,
                   'C':protocol['probe_C'],'ridge_alpha':protocol['ridge_alpha'],'test_evaluated':False}
        path=EXP/'bert_svd_protocol.json'
        if path.exists() and json.loads(path.read_text())!=amendment:raise ValueError('BERT-SVD protocol changed')
        save_json(path,amendment)
    expected_layers=list(layers)
    if args.layers is not None:
        layers=[int(x) for x in args.layers.split(',')]
    modes=protocol['bert_pools'] if args.encoder=='bert' else [protocol['qwen_pool']]
    tr,va=load_split('train'),load_split('valid')
    vv=views(va)
    labels=np.sign(tr['y']).astype(int)+1
    started=time.time()
    rows=[]
    for layer in layers:
        for mode in modes:
            prefix=f'layer{layer}_{mode}'
            finished=out/f'{prefix}_complete.json'
            if finished.exists():
                rows.extend(json.loads(finished.read_text())['rows'])
                continue
            xt=pooled(args.encoder,'train',layer,tr['obs'][:,0],mode)
            xv=[pooled(args.encoder,'valid',layer,d['obs'][:,0],mode) for d in vv]
            scaler=StandardScaler().fit(xt)
            xt=scaler.transform(xt)
            xv=[scaler.transform(x) for x in xv]
            reducer=None
            if args.encoder=='bert_svd':
                reducer=TruncatedSVD(n_components=384,random_state=7,n_iter=5).fit(xt)
                xt=reducer.transform(xt);xv=[reducer.transform(x) for x in xv]
            ridge_trials=[]
            for a in protocol['ridge_alpha']:
                reg=Ridge(alpha=a).fit(xt,tr['y'])
                intensity=[np.clip(reg.predict(x),-3,3) for x in xv]
                ridge_trials.append((float(np.mean([np.abs(va['y']-p).mean() for p in intensity])),a,reg,intensity))
            _,alpha,reg,intensity=min(ridge_trials,key=lambda r:r[0])
            part=[]
            for c in protocol['probe_C']:
                name=f'{prefix}_C{c}'
                clf=LogisticRegression(C=c,class_weight='balanced',max_iter=3000).fit(xt,labels)
                probs=[clf.predict_proba(x) for x in xv]
                ms=[metrics(va['y'],p,r) for p,r in zip(probs,intensity)]
                row={'name':name,'encoder':args.encoder,'layer':layer,'pooling':mode,'C':c,'ridge_alpha':alpha,
                     'primary_score':score_views(va['y'],probs),'clean':ms[0],'mixed40':ms[1],
                     'fit_splits':['train'],'selection_splits':['valid']}
                if reducer is not None:row.update(svd_components=384,svd_seed=7,svd_iterations=5)
                head={'scaler':scaler,'classifier':clf,'ridge':reg,'metadata':row}
                if reducer is not None:head['svd']=reducer
                joblib.dump(head,out/f'{name}.joblib')
                np.savez_compressed(out/f'{name}_valid.npz',ids=va['ids'],probabilities=probs,intensity=intensity)
                part.append(row)
                print(f'{args.encoder}/{name}: mean F1={row["primary_score"]:.6f}, clean={ms[0]["macro_f1"]:.6f}',flush=True)
            save_json(finished,{'rows':part})
            rows.extend(part)
            save_json(out/'search.json',rows)
    best=max(rows,key=lambda r:r['primary_score'])
    eligible=best['clean']['macro_f1']>=protocol['expert_gates']['enter_fusion_clean_valid_f1']
    save_json(out/'selection.json',{'selected':best,'fusion_gate_passed':eligible,'test_evaluated':False,
                                 'encoder_manifest_sha256':digest(EXP/'features'/('bert' if args.encoder=='bert_svd' else args.encoder)/'encoder.json')})
    save_json(out/'complete.json',{'candidate_count':len(rows),'layers':layers,'seconds_this_session':time.time()-started,
                                 'full_prespecified_grid':layers==expected_layers})


if __name__=='__main__':
    main()
