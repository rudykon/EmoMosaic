"""Validation exploration of reliability-gated experts with existing RM core.

Preliminary results only: does not freeze the full experiment or open test data.
The final selector must also compare all retrained RM ensembles.
"""
import argparse
import json

import numpy as np

from data import load_split
from improvement_common import (EXP, initialize_protocol, views, fusion, select_bias, apply_bias,
                                metrics, save_json, score_views, text_observed_fraction)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--encoder',choices=['bert','bert_svd','qwen','qwen7','qwen35_08','qwen35_2','qwen35_4'],required=True)
    args=ap.parse_args()
    protocol=initialize_protocol()
    base=EXP/'probes'/args.encoder
    selection=json.loads((base/'selection.json').read_text())
    expert=selection['selected']
    out=EXP/'preliminary_fusion'/args.encoder
    out.mkdir(parents=True,exist_ok=True)
    if not args.encoder.startswith('bert') and not selection['fusion_gate_passed']:
        save_json(out/'decision.json',{'gate_passed':False,'expert':expert,'decision':'Stop this model before fusion per preregistered threshold',
                                       'test_evaluated':False})
        return
    va=load_split('valid');vv=views(va)
    with np.load(base/f'{expert["name"]}_valid.npz') as z:
        ep,er=z['probabilities'],z['intensity'];assert np.array_equal(z['ids'],va['ids'])
    with np.load(EXP/'existing/rm_core_valid.npz') as z:
        cp,cr=z['probabilities'],z['intensity'];assert np.array_equal(z['ids'],va['ids'])
    fraction=[text_observed_fraction(v) for v in vv]
    trials=[]
    for k in protocol['fusion_exponents']:
        for w in protocol['fusion_weights']:
            p=[fusion(e,c,f,w,k)[0] for e,c,f in zip(ep,cp,fraction)]
            bias=select_bias(va['y'],p)
            trials.append(dict(bias,w0=w,k=k))
    best=max(trials,key=lambda v:v['score'])
    p=[fusion(e,c,f,best['w0'],best['k'],best['bias'])[0] for e,c,f in zip(ep,cp,fraction)]
    reg_trials=[]
    for w in protocol['fusion_weights']:
        weights=[w*f**best['k'] for f in fraction]
        intensity=[a*e+(1-a)*c for a,e,c in zip(weights,er,cr)]
        reg_trials.append((float(np.mean([np.abs(va['y']-r).mean() for r in intensity])),w,intensity))
    mae,reg_w,r=min(reg_trials,key=lambda v:v[0])
    np.savez_compressed(out/'valid_predictions.npz',ids=va['ids'],probabilities=p,intensity=r)
    result={'expert':expert,'core':'original_5seed_rmnet','classification':best,'regression_w0':reg_w,
            'regression_k':best['k'],'regression_valid_mae':mae,'primary_score':score_views(va['y'],p),
            'clean':metrics(va['y'],p[0],r[0]),'mixed40':metrics(va['y'],p[1],r[1]),
            'scope':'preliminary_existing_core_only','test_evaluated':False,
            'reliability':'observed informative BERT content slots / all content slots; UNK excluded'}
    save_json(out/'selection.json',result)
    save_json(out/'search.json',trials)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
