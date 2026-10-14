import numpy as np


def classes(y, thresholds=(0.,0.)):
    return np.where(y < -thresholds[0],0,np.where(y > thresholds[1],2,1))


def classification(y, pred):
    cm=np.bincount(classes(y)*3+pred,minlength=9).reshape(3,3)
    support=cm.sum(1); diag=cm.diagonal()
    f1=2*diag/np.maximum(cm.sum(0)+support,1)
    return {'accuracy':float(diag.sum()/max(1,cm.sum())), 'macro_f1':float(f1.mean()),
            'weighted_f1':float((f1*support).sum()/max(1,cm.sum())),
            'neutral_recall':float(diag[1]/max(1,support[1])), 'confusion':cm.tolist()}


def metrics(y, pred, thresholds=(.2,.2)):
    y=np.asarray(y);pred=np.asarray(pred)
    r=classification(y,classes(pred,thresholds))
    r['mae']=float(np.abs(pred-y).mean())
    # A constant prediction has undefined Pearson r. Store 0 for scoring, flag it.
    defined=bool(np.std(pred)>1e-7 and np.std(y)>1e-7)
    r['corr_defined']=defined
    r['corr']=float(np.corrcoef(y,pred)[0,1]) if defined else 0.
    r['selection_score']=r['corr']-r['mae']+r['macro_f1']
    return r


def tune_thresholds(y, pred, step=.05, limit=1.):
    best=(-float('inf'),None)
    for neg in np.arange(0,limit+step/2,step):
        for pos in np.arange(0,limit+step/2,step):
            score=classification(y,classes(pred,(neg,pos)))['macro_f1']
            if score>best[0]+1e-12:best=(score,(float(neg),float(pos)))
    return best[1]
