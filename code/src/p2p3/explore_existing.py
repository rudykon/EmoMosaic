"""Stages 0/1: existing RM-Net bias, text/SVD probes, and probability ensemble.

Only train and valid caches are opened. Final test evaluation is a separate step.
"""
from __future__ import annotations

import argparse
import csv
import time

import joblib
import numpy as np
import torch
from scipy.special import softmax
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from boosted_classifier import sequence_features
from compare_boosted import rm_predict
from data import load_split, ROOT
from improvement_common import (EXP, initialize_protocol, save_json, views, pool_tokens,
                                metrics, score_views, select_bias, apply_bias, digest)
from predict import Predictor
from train import device_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    protocol = initialize_protocol()
    out = EXP / 'existing'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        print('Existing-assets exploration already complete', flush=True)
        return
    start = time.time()
    train, valid = load_split('train'), load_split('valid')
    vv = views(valid)
    y, labels = valid['y'], np.sign(train['y']).astype(int) + 1
    rows, predictions = [], {}

    def record(name, probs, intensity=None, params=None):
        ms = [metrics(y, p, intensity[i] if intensity is not None else None) for i,p in enumerate(probs)]
        row = {'name': name, 'primary_score': score_views(y, probs),
               'clean': ms[0], 'mixed40': ms[1], 'parameters': params or {}}
        rows.append(row)
        predictions[name] = probs
        np.savez_compressed(out / f'{name}_valid.npz', ids=valid['ids'], probabilities=np.asarray(probs),
                            **({'intensity': np.asarray(intensity)} if intensity is not None else {}))
        save_json(out / 'search.json', rows)
        print(f'{name}: valid mean F1={row["primary_score"]:.6f}', flush=True)

    rm_cache = out / 'rm_core_valid.npz'
    if rm_cache.exists():
        with np.load(rm_cache) as z:
            assert np.array_equal(z['ids'], valid['ids'])
            rm_probs, rm_y = z['probabilities'], z['intensity']
    else:
        predictor = Predictor()
        tensors = device_data(valid, 'cpu')
        parts = [rm_predict(predictor, tensors, d['obs']) for d in vv]
        rm_probs = np.asarray([p['aux_probs'] for p in parts])
        rm_y = np.asarray([p['intensity'] for p in parts])
        np.savez_compressed(rm_cache, ids=valid['ids'], probabilities=rm_probs, intensity=rm_y)
    record('rm_auxiliary', rm_probs, rm_y)
    bias = select_bias(y, rm_probs)
    record('rm_auxiliary_bias', [apply_bias(p, bias['bias']) for p in rm_probs], rm_y, bias)

    # Exact existing SVD family ablation: same C/seed, with and without A/V.
    feats_tr = sequence_features(train)
    feats_va = [sequence_features(d) for d in vv]
    for kind, width in [('all', feats_tr.shape[1]), ('text', 4 * train['text'].shape[-1])]:
        svd = TruncatedSVD(n_components=384, random_state=7, n_iter=5)
        xt = svd.fit_transform(feats_tr[:, :width])
        xv = [svd.transform(v[:, :width]) for v in feats_va]
        clf = LogisticRegression(C=.003, solver='saga', max_iter=250, random_state=7).fit(xt, labels)
        probs = [clf.predict_proba(x) for x in xv]
        offset = select_bias(y, probs)
        joblib.dump({'svd': svd, 'classifier': clf, 'bias': np.asarray(offset['bias']), 'width': width}, out / f'svd_{kind}.joblib')
        record(f'svd_{kind}_raw', probs, params={'C': .003, 'seed': 7})
        record(f'svd_{kind}_bias', [apply_bias(p,offset['bias']) for p in probs], params=offset)

    # Text-only linear baseline, exactly the same probe used for new encoders.
    for pooling in protocol['bert_pools']:
        xtr = pool_tokens(train['text'], train['obs'][:,0], pooling)
        xva = [pool_tokens(d['text'], d['obs'][:,0], pooling) for d in vv]
        scaler = StandardScaler().fit(xtr)
        xt, xv = scaler.transform(xtr), [scaler.transform(x) for x in xva]
        ridge_trials = []
        for alpha in protocol['ridge_alpha']:
            ridge = Ridge(alpha=alpha).fit(xt, train['y'])
            reg = [np.clip(ridge.predict(x), -3, 3) for x in xv]
            ridge_trials.append((float(np.mean([np.abs(y-r).mean() for r in reg])),alpha,ridge,reg))
        _, alpha, ridge, reg = min(ridge_trials, key=lambda z: z[0])
        for c in protocol['probe_C']:
            name = f'bert_last_{pooling}_C{c}'
            clf = LogisticRegression(C=c, class_weight='balanced', max_iter=3000).fit(xt, labels)
            probs = [clf.predict_proba(x) for x in xv]
            joblib.dump({'scaler':scaler,'classifier':clf,'ridge':ridge,'pooling':pooling}, out / f'{name}.joblib')
            record(name, probs, reg, {'C':c,'ridge_alpha':alpha,'class_weight':'balanced'})

    # Different-family diversity; bias selected on average F1 of the two views.
    fusion_trials = []
    svd_probs = predictions['svd_all_raw']
    for weight in protocol['fusion_weights']:
        probs = [weight*s+(1-weight)*r for s,r in zip(svd_probs,rm_probs)]
        b = select_bias(y, probs)
        fusion_trials.append(dict(b, svd_weight=weight))
    choice = max(fusion_trials, key=lambda x:x['score'])
    probs = [apply_bias(choice['svd_weight']*s+(1-choice['svd_weight'])*r,choice['bias'])
             for s,r in zip(svd_probs,rm_probs)]
    record('rm_svd_fusion_bias', probs, rm_y, choice)
    save_json(out / 'fusion_search.json', fusion_trials)
    best = max(rows, key=lambda r:r['primary_score'])
    save_json(out / 'selection.json', {'selected':best,'scope':'stage0_and_existing_assets_only',
                                     'test_evaluated':False,'protocol_sha256':digest(EXP/'protocol.json')})
    with (out / 'validation_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name','primary_score','clean_f1','missing_f1','clean_accuracy','missing_accuracy'])
        writer.writeheader()
        for row in rows:
            writer.writerow({'name':row['name'],'primary_score':row['primary_score'],
                             'clean_f1':row['clean']['macro_f1'],'missing_f1':row['mixed40']['macro_f1'],
                             'clean_accuracy':row['clean']['accuracy'],'missing_accuracy':row['mixed40']['accuracy']})
    save_json(out / 'complete.json', {'seconds':time.time()-start,'candidate_count':len(rows),
                                    'test_loaded':False,'selected_name':best['name'],
                                    'input_sha256':{s:digest(ROOT/f'outputs/problem2/cache/{s}.npz') for s in ['train','valid']}})


if __name__ == '__main__':
    main()
