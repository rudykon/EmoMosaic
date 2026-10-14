"""Five-fold train-only video-group CV for a fixed validation-selected probe.

This measures conditional stability, not nested-CV generalization: the layer and
regularization were selected using models trained on all of train. No fold can
share an original video's clips with its fitting subset.
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from data import load_split
from improvement_common import EXP, save_json, metrics
from probe_text_experts import pooled
from train import validation_masks


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--encoder',choices=['bert','bert_svd','qwen','qwen7','qwen35_08','qwen35_2','qwen35_4'],required=True)
    args=ap.parse_args()
    out=EXP/'probes'/args.encoder
    selected=json.loads((out/'selection.json').read_text())['selected']
    d=load_split('train')
    y=d['y'];labels=np.sign(y).astype(int)+1
    groups=np.asarray([str(i).split('$_$')[0] for i in d['ids']])
    clean=pooled(args.encoder,'train',selected['layer'],d['obs'][:,0],selected['pooling'])
    mask=validation_masks(d['lengths'],20260923)[1]
    corrupt=pooled(args.encoder,'train',selected['layer'],(d['obs']&mask)[:,0],selected['pooling'])
    folds=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20260926)
    prob=np.zeros((2,len(y),3));intensity=np.zeros((2,len(y)));fold_id=np.full(len(y),-1)
    results=[]
    for fold,(fit,holdout) in enumerate(folds.split(clean,labels,groups)):
        assert not set(groups[fit])&set(groups[holdout])
        scaler=StandardScaler().fit(clean[fit]);xfit=scaler.transform(clean[fit])
        reducer=None
        if selected.get('svd_components'):
            reducer=TruncatedSVD(n_components=selected['svd_components'],random_state=selected['svd_seed'],
                                 n_iter=selected['svd_iterations']).fit(xfit)
            xfit=reducer.transform(xfit)
        clf=LogisticRegression(C=selected['C'],class_weight='balanced',max_iter=3000).fit(xfit,labels[fit])
        ridge=Ridge(alpha=selected['ridge_alpha']).fit(xfit,y[fit])
        scores=[]
        for j,x in enumerate([clean,corrupt]):
            xx=scaler.transform(x[holdout])
            if reducer is not None:xx=reducer.transform(xx)
            p=clf.predict_proba(xx);r=np.clip(ridge.predict(xx),-3,3)
            prob[j,holdout]=p;intensity[j,holdout]=r
            scores.append(metrics(y[holdout],p,r))
        fold_id[holdout]=fold
        result={'fold':fold,'fit_n':len(fit),'holdout_n':len(holdout),'fit_videos':len(set(groups[fit])),
                'holdout_videos':len(set(groups[holdout])),'video_overlap':0,'clean':scores[0],'mixed40':scores[1],
                'primary_score':float(np.mean([m['macro_f1'] for m in scores]))}
        results.append(result)
        print(f'{args.encoder} fold {fold}: mean F1={result["primary_score"]:.6f}',flush=True)
    assert np.all(fold_id>=0)
    np.savez_compressed(out/'train_group_oof.npz',ids=d['ids'],groups=groups,fold_id=fold_id,
                        probabilities=prob,intensity=intensity)
    save_json(out/'train_group_cv.json',{'selected_configuration':selected,'folds':results,
                                       'mean_primary':float(np.mean([r['primary_score'] for r in results])),
                                       'std_primary':float(np.std([r['primary_score'] for r in results],ddof=1)),
                                       'interpretation':'Conditional stability of fixed validation-selected configuration; not nested unbiased performance.'})


if __name__=='__main__':
    main()
