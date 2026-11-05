"""Rebuild fixed-shape interfaces from untruncated, independently extracted features."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def block_pool(x, stride, valid=None):
    values=[];masks=[];ranges=[]
    for start in range(0,len(x),stride):
        end=min(start+stride,len(x));ids=np.arange(start,end)
        if valid is not None:ids=ids[valid[ids]]
        values.append(x[ids].mean(axis=0) if len(ids) else np.zeros(x.shape[1]))
        masks.append(bool(len(ids)));ranges.append([start,end])
    return np.asarray(values,np.float32),np.array(masks),ranges

def export(out):
    m=pd.read_csv(out/'manifest.csv',keep_default_na=False)
    assert len(m)==100 and (m.status=='complete').all()
    base=dict(id=m.sample_id.tolist(),raw_text=m.text.to_numpy(),annotations=m.annotation.to_numpy(),
        classification_labels=m.annotation.map({'Negative':0,'Neutral':1,'Positive':2}).to_numpy(np.int64),
        regression_labels=m.label.to_numpy(np.float32),
        text_lengths=np.minimum(m.n_bert_tokens.to_numpy(np.int32),50),
        original_text_lengths=m.n_bert_tokens.to_numpy(np.int32),
        word_lengths=m.n_words.to_numpy(np.int32),
        feature_version='p1_raw_v1_not_equivalent_to_attachment2',
        text=np.zeros((100,50,768),np.float16),text_bert=np.zeros((100,3,50),np.int64),
        token_word_index=np.full((100,50),-1,np.int32))
    aligned={**base,'audio':np.zeros((100,50,74),np.float32),'vision':np.zeros((100,50,35),np.float32),
             'mask':np.zeros((100,50,3),bool)}
    unaligned={**base,'audio':np.zeros((100,500,74),np.float32),'vision':np.zeros((100,500,35),np.float32),
        'audio_mask':np.zeros((100,500),bool),'vision_mask':np.zeros((100,500),bool),
        'audio_lengths':np.zeros(100,np.int32),'vision_lengths':np.zeros(100,np.int32),
        'text_mask':np.zeros((100,50),bool)}
    ra=int(np.ceil(m.n_audio.max()/500));rv=int(np.ceil(m.n_vision.max()/500))
    mapping={}
    for j,row in m.iterrows():
        sid=row.sample_id;t=np.load(out/'L0_frame'/f'{sid}.text.npz');l=np.load(out/'L1_word'/f'{sid}.npz')
        a=np.load(out/'L0_frame'/f'{sid}.audio.npz');v=np.load(out/'L0_frame'/f'{sid}.vision.npz')
        n=len(t['token_ids']);keep=np.arange(n) if n<=50 else np.r_[np.arange(49),n-1]
        size=len(keep);wi=t['token_word'][keep]
        aligned['text'][j,:size]=t['token_hidden'][keep]
        aligned['text_bert'][j,0,:size]=t['token_ids'][keep]
        aligned['text_bert'][j,1,:size]=1
        aligned['token_word_index'][j,:size]=wi
        aligned['mask'][j,:size,0]=1
        for pos,word in enumerate(wi):
            if word>=0:
                aligned['audio'][j,pos]=l['audio'][word];aligned['vision'][j,pos]=l['vision'][word]
                aligned['mask'][j,pos,1:]=l['mask'][word,1:]
        aa,am,ar=block_pool(a['audio'],ra)
        vv,vm,vr=block_pool(v['vision'],rv,v['face_mask'])
        unaligned['audio'][j,:len(aa)]=aa;unaligned['vision'][j,:len(vv)]=vv
        unaligned['audio_lengths'][j]=len(aa);unaligned['vision_lengths'][j]=len(vv)
        unaligned['audio_mask'][j,:len(aa)]=am;unaligned['vision_mask'][j,:len(vv)]=vm
        unaligned['text_mask'][j,:size]=True
        mapping[sid]=dict(source_token_positions=keep.tolist(),audio_frame_ranges=ar,video_frame_ranges=vr)
    for filename,data in [('P1_aligned_50.pkl',aligned),('P1_unaligned_50.pkl',unaligned)]:
        with (out/filename).open('wb') as f:pickle.dump({'p1':data},f,protocol=4)
    save_json(out/'interface_mapping.json',dict(audio_stride=ra,vision_stride=rv,
        padding='right zero; use explicit masks; valid zero values are not missing',
        special_tokens='CLS/SEP text valid, audio/vision invalid; truncated sequences retain final SEP',
        samples=mapping))
    stats={}
    for modality,dims in [('audio',74),('vision',35),('text',768)]:
        arrays=[]
        for sid in m.sample_id:
            file=out/'L0_frame'/f'{sid}.{modality}.npz'
            z=np.load(file);x=z[modality].astype(np.float64)
            if modality=='vision':x=x[z['face_mask']]
            arrays.append(x)
        x=np.concatenate(arrays);stats[modality]=dict(count=len(x),mean=x.mean(axis=0).tolist(),std=x.std(axis=0).tolist())
    save_json(out/'stats.json',dict(usage='Descriptive QC of attachment1 only; do not use as normalization for problems2/3.',modalities=stats))
    print('exported',ra,rv,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/problem1');a=p.parse_args()
    export(ROOT/a.output)
