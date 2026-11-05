"""Train-only normalization of the 13 frozen BERT layers for scalar mixing."""
import json
from pathlib import Path

import numpy as np

from data import load_split
from improvement_common import EXP, digest, save_json


def prepare_stats(folder=None):
    folder=Path(folder) if folder else EXP/'features/bert'
    path=folder/'norm_stats.json'
    if path.exists():
        return json.loads(path.read_text())
    if not (folder/'train/complete.json').exists():
        raise RuntimeError('Train BERT extraction incomplete')
    hidden=np.load(folder/'train/hidden.npy',mmap_mode='r')
    obs=load_split('train')['obs'][:,0]
    count=int(obs.sum())
    sums=np.zeros((hidden.shape[1],hidden.shape[-1]),np.float64)
    squares=np.zeros_like(sums)
    for s in range(0,len(obs),32):
        x=np.asarray(hidden[s:s+32],np.float64)
        mask=obs[s:s+32,None,:,None]
        sums+=(x*mask).sum((0,2))
        squares+=(x*x*mask).sum((0,2))
    mean=sums/count
    std=np.maximum(np.sqrt(np.maximum(squares/count-mean*mean,0)),1e-5)
    result={'mean':mean.tolist(),'std':std.tolist(),'n_observed':count,'fit_split':'train',
            'hidden_sha256':json.loads((folder/'train/complete.json').read_text())['hidden_sha256']}
    save_json(path,result)
    return result


def load_layer_split(split,folder=None):
    folder=Path(folder) if folder else EXP/'features/bert'
    if not (folder/split/'complete.json').exists():
        raise RuntimeError(f'{split} BERT extraction incomplete')
    data=load_split(split)
    stats=prepare_stats(folder)
    mean=np.asarray(stats['mean'],np.float32)[None,:,None,:]
    std=np.asarray(stats['std'],np.float32)[None,:,None,:]
    hidden=np.load(folder/split/'hidden.npy',mmap_mode='r')
    text=np.empty(hidden.shape,np.float32)
    for start in range(0,len(hidden),32):
        stop=start+32
        text[start:stop]=np.clip((hidden[start:stop].astype(np.float32)-mean)/std,-5,5)*data['obs'][start:stop,0,None,:,None]
    data['text']=text
    return data
