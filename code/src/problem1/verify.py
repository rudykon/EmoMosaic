"""Verify every source/feature/index/mask, by reconstruction rather than file presence."""
import argparse
import itertools
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import av
from core import ROOT, PROJECT, DATA, records, digest, pool_words, ctc_viterbi, save_json
from export import block_pool

def check_ctc():
    # Exhaustively compare exact CTC DP to all alignments of repeated target A A.
    p=np.array([[.1,.9],[.6,.4],[.8,.2],[.1,.9],[.7,.3]],np.float32)
    target=[1,1];best=-np.inf
    for path in itertools.product([0,1],repeat=len(p)):
        collapsed=[a for i,a in enumerate(path) if a and (i==0 or a!=path[i-1])]
        if collapsed==target:best=max(best,sum(np.log(p[i,a]) for i,a in enumerate(path)))
    states=ctc_viterbi(np.log(p),target,0);labels=np.array([0,1,0,1,0])[states]
    assert np.isclose(np.log(p[np.arange(len(p)),labels]).sum(),best)

def verify(out):
    check_ctc();original=records();m=pd.read_csv(out/'manifest.csv',keep_default_na=False)
    assert len(m)==100 and m.sample_id.tolist()==[r['sample_id'] for r in original]
    assert (m.status=='complete').all()
    assert digest(DATA/'label-100.xlsx')==json.loads((out/'source_labels.json').read_text())['sha256']
    aligned=pickle.load((out/'P1_aligned_50.pkl').open('rb'))['p1']
    unaligned=pickle.load((out/'P1_unaligned_50.pkl').open('rb'))['p1']
    maps=json.loads((out/'interface_mapping.json').read_text())
    counts=dict(samples=0,words=0,audio_frames=0,video_frames=0,source_hashes=0,reconstructed_words=0)
    for j,r in enumerate(original):
        sid=r['sample_id'];row=m.iloc[j]
        assert digest(PROJECT/r['path'])==row.source_sha256
        assert digest(PROJECT/r['path'],'md5')==row.source_md5
        assert row.text==r['text'] and abs(row.label-r['label'])<1e-12 and row.annotation==r['annotation']
        a=np.load(out/'L0_frame'/f'{sid}.audio.npz');v=np.load(out/'L0_frame'/f'{sid}.vision.npz')
        t=np.load(out/'L0_frame'/f'{sid}.text.npz');l=np.load(out/'L1_word'/f'{sid}.npz')
        words=json.loads((out/'word_times'/f'{sid}.json').read_text())
        for z in [a,v,t,l]:
            for k in z.files:assert np.isfinite(z[k]).all(),(sid,k)
        n=len(words);assert n==len(l['text'])==len(l['audio'])==len(l['vision'])==row.n_words
        assert a['audio'].shape==(row.n_audio,74) and v['vision'].shape==(row.n_vision,35)
        assert np.all(np.diff(a['audio_times'])>0) and np.all(np.diff(v['vision_times'])>0)
        with av.open(str(PROJECT/r['path'])) as container:
            source_pts=np.array([float(f.pts*f.time_base)-row.common_start for f in container.decode(video=0)])
        indices=np.abs(source_pts[:,None]-v['vision_times'][None,:]).argmin(axis=0)
        np.testing.assert_array_equal(v['source_video_index'],indices)
        np.testing.assert_allclose(v['source_video_pts'],source_pts[indices],atol=1e-12)
        # Source frame gaps can exceed the nominal FPS. Bound the measured error
        # by one output grid step, and verify against actual decoded source PTS.
        assert np.max(np.abs(v['source_video_pts']-v['vision_times']))<=.040001
        assert np.all(v['vision'][~v['face_mask']]==0)
        end=0
        for w in words:
            assert w['timing_usable']==bool(w['confidence']>=.5 and row.align_conf>=.5 and row.wer_ctc<=.5)
            assert 0<=end<=w['s']<w['e']<=row.duration+1e-6,(sid,w)
            end=w['e'];assert r['text'][w['char_start']:w['char_end']]==w['word']
            # Independent oracle: derive endpoints from CTC indices, not pool_words.
            start_sample=320*w['ctc_frames'][0]+40
            end_sample=min(320*(w['ctc_frames'][1]-1)+360,int(row.decoded_audio_samples))
            assert w['s_sample']==start_sample and w['e_sample']==end_sample
            assert w['s']==start_sample/16000 and w['e']==end_sample/16000
            for field,centers in [('a_frames',160*np.arange(len(a['audio']))+200),
                                  ('v_frames',640*np.arange(len(v['vision'])))]:
                expected=np.flatnonzero((centers>=start_sample)&(centers<end_sample))
                if not len(expected):
                    expected=np.array([np.abs(2*centers-start_sample-end_sample).argmin()])
                np.testing.assert_array_equal(w[field],expected,err_msg=f'{sid}: {w["word"]}: {field}')
            assert w['source_video_frames']==v['source_video_index'][w['v_frames']].tolist()
            expected=t['token_hidden'][w['bert_subtokens']].astype(np.float32).mean(axis=0)
            assert np.max(np.abs(expected-l['text'][w['i']]))<.004
        old=json.dumps(words,sort_keys=True)
        reconstructed=pool_words(words,a['audio'],a['audio_times'],v['vision'],v['vision_times'],v['face_mask'])
        assert json.dumps(words,sort_keys=True)==old
        for k,val in reconstructed.items():np.testing.assert_array_equal(l[k],val)
        selected=maps['samples'][sid]['source_token_positions'];size=len(selected)
        assert aligned['text_lengths'][j]==size
        np.testing.assert_array_equal(aligned['text'][j,:size],t['token_hidden'][selected])
        np.testing.assert_array_equal(aligned['text_bert'][j,0,:size],t['token_ids'][selected])
        assert aligned['text_bert'][j,0,size-1]==102
        for pos,w in enumerate(t['token_word'][selected]):
            if w>=0:
                for k in ['audio','vision']:np.testing.assert_array_equal(aligned[k][j,pos],l[k][w])
            else:assert not aligned['mask'][j,pos,1:].any()
        for k in ['text','audio','vision']:assert np.all(aligned[k][j,size:]==0)
        for k,z,valid,stride in [('audio',a,None,maps['audio_stride']),('vision',v,v['face_mask'],maps['vision_stride'])]:
            pooled,mask,ranges=block_pool(z[k],stride,valid)
            size=len(pooled);assert unaligned[k+'_lengths'][j]==size
            np.testing.assert_array_equal(unaligned[k][j,:size],pooled)
            np.testing.assert_array_equal(unaligned[k+'_mask'][j,:size],mask)
            assert np.all(unaligned[k][j,size:]==0)
        counts['samples']+=1;counts['source_hashes']+=1;counts['words']+=n;counts['reconstructed_words']+=n
        counts['audio_frames']+=len(a['audio']);counts['video_frames']+=len(v['vision'])
    for data in [aligned,unaligned]:
        assert data['id']==m.sample_id.tolist()
        for k,v in data.items():
            if isinstance(v,np.ndarray) and v.dtype.kind in 'fbiu':assert np.isfinite(v).all()
    save_json(out/'verification.json',dict(status='passed',checks=counts,
        scope=['original media and labels unchanged','100/100 complete','all arrays finite',
        'word boundaries monotonic and in media bounds','every word rebuilt from source-frame features',
        'all frame indices and original text slices checked','all fixed tensors rebuilt and padding checked',
            'exact CTC algorithm checked against exhaustive repeated-character case',
            'half-open frame membership independently checked on integer sample clock']))
    print('PASS',counts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/problem1');args=p.parse_args();verify(ROOT/args.output)
