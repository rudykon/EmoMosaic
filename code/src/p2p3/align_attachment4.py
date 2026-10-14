"""Run with .venv-p1: independent raw-media alignment, never model features."""
import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/problem1'))
from core import (TextExtractor,audio_features,probe,waveform,vision_features,
                  pool_words,digest,save_json)
import numpy as np
from data import load_pickle,unwrap
from project_paths import relative_asset
from p3_common import OUT3,attachment_paths,token_mapping


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--force',action='store_true');args=ap.parse_args()
    out=OUT3/'alignment';out.mkdir(parents=True,exist_ok=True)
    engine=None;summary=[]
    for path in attachment_paths():
        video=path.parent/'videos'/f'{path.stem}.mp4';dest=out/f'{path.stem}.json'
        source_hash=digest(path);video_hash=digest(video)
        if dest.exists() and (out/f'{path.stem}.npz').exists() and not args.force:
            r=json.loads(dest.read_text())
            assert r['feature_sha256']==source_hash and r['video_sha256']==video_hash
            summary.append({k:v for k,v in r.items() if k not in ['words','positions']});continue
        start=time.monotonic();d=unwrap(load_pickle(path));raw=str(d['raw_text'])
        meta=probe(video);y=waveform(video,meta);audio,at,audit=audio_features(y)
        meta['decoded_audio_samples']=len(y)
        if engine is None:engine=TextExtractor(threads=4)
        words,text,quality=engine.extract(raw,y)
        visual,vaudit=vision_features(video,meta)
        for w in words:
            w['timing_usable']=bool(w['confidence']>=.5 and quality['align_conf']>=.5 and quality['wer_ctc']<=.5)
        pool_words(words,audio,at,visual['vision'],visual['vision_times'],visual['face_mask'])
        ids=np.asarray(d['text_bert'])[0].astype(int)
        positions=token_mapping(raw,ids,engine.tokenizer)
        mismatch=[]
        for pos in positions:
            w=next((w for w in words if pos['word_char_start']==w['char_start'] and pos['word_char_end']==w['char_end']),None)
            if w is None:
                pos.update(time_start=None,time_end=None,timing_usable=False,quality_flag='special_or_unaligned',keyframe=None)
                continue
            vi=np.asarray(w['v_frames']);valid=vi[visual['face_mask'][vi]]
            midpoint_twice=w['s_sample']+w['e_sample']
            if len(valid):
                ticks=np.rint(visual['vision_times'][valid]*16000).astype(np.int64)
                key=int(valid[np.argmin(abs(2*ticks-midpoint_twice))]);face=True
            else:
                key=int(vi[0]);face=False
            frame=dict(frame_index=int(visual['source_video_index'][key]),
                pts_relative=float(visual['source_video_pts'][key]),
                pts_absolute=float(visual['source_video_pts'][key]+meta['common_start']),
                grid_index=key,face=face,
                endpoint_frames=[int(visual['source_video_index'][vi[0]]),int(visual['source_video_index'][vi[-1]])])
            pos.update(time_start=w['s'],time_end=w['e'],confidence=w['confidence'],
                timing_usable=w['timing_usable'],quality_flag='estimated' if w['timing_usable'] else 'candidate_low_trust',
                a_frames=w['a_frames'],v_frames=w['v_frames'],keyframe=frame)
            official_zero=not np.any(np.asarray(d['vision'])[pos['position']])
            mismatch.append(dict(position=pos['position'],official_vision_zero=official_zero,
                independent_no_face=not face,agree=official_zero==(not face),timing_usable=w['timing_usable']))
        flags=[]
        if quality['wer_ctc']>.5:flags.append('sample_WER_above_0.5')
        if quality['align_conf']<.5:flags.append('sample_CTC_below_0.5')
        face_rate=float(visual['face_mask'].mean())
        if face_rate<.2:flags.append('low_face_rate')
        result=dict(id=str(d['id']),file=path.name,raw_text=raw,feature_sha256=source_hash,
            video_sha256=video_hash,video=relative_asset(video),stream=meta,
            **quality,face_rate=face_rate,flags=flags,words=words,positions=positions,
            zero_face_comparison=mismatch,observed_zero_face_agreement=float(np.mean([r['agree'] for r in mismatch])) if mismatch else None,
            timing_usable_fraction=float(np.mean([w['timing_usable'] for w in words])),
            limitation='Independent CTC word timing; not organizer alignment ground truth. F0/RMS and face detection are descriptive replay only.',
            seconds=time.monotonic()-start)
        np.savez_compressed(out/f'{path.stem}.npz',audio_times=at,f0=audio[:,0],rms=audio[:,4],
            vision_times=visual['vision_times'],face_mask=visual['face_mask'],
            source_video_pts_relative=visual['source_video_pts'],source_video_index=visual['source_video_index'])
        save_json(dest,result);summary.append({k:v for k,v in result.items() if k not in ['words','positions']})
        print(path.stem,'aligned',len(words),'words',round(result['seconds'],1),'s',flags,flush=True)
    save_json(out/'summary.json',summary)
    save_json(out/'environment.json',dict(python=platform.python_version(),
        packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','mediapipe','av','opensmile','pyworld','librosa']},
        protocol_sha256=digest(ROOT/'src/problem1/config.yaml'),
        models=json.loads((ROOT/'src/problem1/model_manifest.json').read_text())))


if __name__=='__main__':main()
