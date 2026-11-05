"""Run the complete 100-sample extraction. All original data remain read-only."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import av
import imageio_ffmpeg

from core import (ROOT, PROJECT, DATA, AUDIO_NAMES, VISION_NAMES, TextExtractor,
    audio_features, digest, pool_words, probe, records, save_json, vision_features, waveform)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='outputs/problem1')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--stage',choices=['all','audio_text','vision','pool'],default='all')
    args=parser.parse_args()
    out=ROOT/args.output
    for name in ['L0_frame','L1_word','word_times','logs','figures','metadata']:(out/name).mkdir(parents=True,exist_ok=True)
    entries=records()
    names=['torch','transformers','numpy','pandas','scipy','av','imageio-ffmpeg','librosa',
           'opensmile','pyworld','pysptk','mediapipe','matplotlib','scikit-learn','openpyxl','num2words']
    save_json(out/'metadata/environment.json',dict(python=platform.python_version(),
        packages={k:importlib.metadata.version(k) for k in names},platform=platform.system(),
        cpu=platform.machine(),cpu_count=os.cpu_count(),torch_threads=4,device='cpu',seed=2026,
        ffmpeg=imageio_ffmpeg.get_ffmpeg_version(),libav=av.library_versions))
    save_json(out/'feature_schema.json',dict(audio=AUDIO_NAMES,vision=VISION_NAMES,
        text='Frozen bert-base-uncased final hidden layer, mean of overlapping WordPieces',
        dimensions={'text':768,'audio':74,'vision':35},
        interval_convention='[start,end); all stored frame lists are zero-based; CTC ranges are exclusive at end'))
    engine=None
    for row in entries[:args.limit]:
        sid=row['sample_id'];p=PROJECT/row['path'];begin=time.monotonic()
        record_file=out/'logs'/f'{sid}.json'
        meta_path=out/'logs'/f'{sid}.probe.json'
        try:
            if meta_path.exists() and not args.force:
                meta=json.loads(meta_path.read_text())
            else:
                meta=probe(p);meta['source_sha256']=digest(p);meta['source_md5']=digest(p,'md5');save_json(meta_path,meta)
            apath=out/'L0_frame'/f'{sid}.audio.npz'
            tpath=out/'L0_frame'/f'{sid}.text.npz'
            wpath=out/'word_times'/f'{sid}.json'
            vpath=out/'L0_frame'/f'{sid}.vision.npz'
            lpath=out/'L1_word'/f'{sid}.npz'
            rec=json.loads(record_file.read_text()) if record_file.exists() else {}
            rec.update(row);rec.update(meta)
            if args.stage in ('all','audio_text') and (args.force or not (apath.exists() and tpath.exists() and wpath.exists())):
                y=waveform(p,meta)
                rec['decoded_audio_samples']=len(y)
                rec['decoded_audio_duration']=len(y)/16000
                audio,at,audit=audio_features(y)
                np.savez_compressed(apath,audio=audio,audio_times=at)
                if engine is None:engine=TextExtractor()
                words,text,align=engine.extract(row['text'],y)
                # float16 only for bounded BERT embeddings; acoustic physical values remain float32.
                quant_error=max(float(np.max(np.abs(text[k]-text[k].astype(np.float16).astype(np.float32)))) for k in ['text','token_hidden'])
                text['text']=text['text'].astype(np.float16);text['token_hidden']=text['token_hidden'].astype(np.float16)
                np.savez_compressed(tpath,**text)
                save_json(wpath,words)
                rec.update(audit);rec.update(align);rec['text_quantization_max_abs']=quant_error
                rec['audio_text_seconds']=time.monotonic()-begin
                print(sid,'audio/text',len(audio),'frames',len(words),'words',flush=True)
            if args.stage in ('all','vision') and (args.force or not vpath.exists()):
                vb=time.monotonic();visual,vaudit=vision_features(p,meta)
                np.savez_compressed(vpath,**visual);rec.update(vaudit)
                rec['vision_seconds']=time.monotonic()-vb
                print(sid,'vision',len(visual['vision']),'frames',flush=True)
            if args.stage in ('all','pool') and all(q.exists() for q in [apath,tpath,wpath,vpath]):
                a=np.load(apath);t=np.load(tpath);v=np.load(vpath);words=json.loads(wpath.read_text())
                if 'decoded_audio_samples' not in rec:
                    rec['decoded_audio_samples']=len(waveform(p,meta))
                    rec['decoded_audio_duration']=rec['decoded_audio_samples']/16000
                for w in words:
                    w['timing_usable']=bool(w['confidence']>=.5 and rec['align_conf']>=.5 and rec['wer_ctc']<=.5)
                pooled=pool_words(words,a['audio'],a['audio_times'],v['vision'],v['vision_times'],v['face_mask'])
                np.savez_compressed(lpath,text=t['text'],**pooled)
                for w in words:
                    w['source_video_frames']=v['source_video_index'][w['v_frames']].tolist()
                    w['source_video_pts']=v['source_video_pts'][w['v_frames']].tolist()
                save_json(wpath,words)
                flags=[]
                if abs(meta['T_video']-meta['T_audio'])>.1:flags.append('av_duration_difference')
                face_rate=float(v['face_mask'].mean())
                if face_rate<.2:flags.append('low_face')
                if rec.get('wer_ctc',0)>.5:flags.append('text_mismatch')
                if rec.get('align_conf',1)<.5:flags.append('low_alignment_confidence')
                n_tokens=len(t['token_ids']);n=len(words)
                if n_tokens>50:flags.append('interface_truncated')
                rec.update(n_audio=len(a['audio']),n_vision=len(v['vision']),n_words=n,n_bert_tokens=n_tokens,
                    L_word=min(n,50),L_aligned=min(n_tokens,50),truncated=n_tokens>50,
                    n_truncated_wordpieces=max(0,n_tokens-50),face_rate=face_rate,
                    word_vision_rate=float(pooled['availability_mask'][:,2].mean()),
                    timing_usable_words=int(pooled['alignment_mask'].sum()),
                    timing_usable_rate=float(pooled['alignment_mask'].mean()),
                    audio_fallbacks=int(pooled['fallback'][:,0].sum()),video_fallbacks=int(pooled['fallback'][:,1].sum()),
                    mean_word_ms=float(np.mean([w['e']-w['s'] for w in words])*1000),
                    flags=';'.join(flags),status='complete',dimensions='768/74/35',
                    granularity='word; audio 10ms; video 40ms',
                    notes='保留原始标签；label-100.xlsx修复说明适用于全体记录')
            save_json(record_file,rec)
        except Exception:
            (out/'logs'/f'{sid}.error.txt').write_text(traceback.format_exc())
            raise
    summary=[]
    for row in entries:
        path=out/'logs'/f"{row['sample_id']}.json"
        if path.exists():summary.append(json.loads(path.read_text()))
    pd.DataFrame(summary).to_csv(out/'manifest.csv',index=False)
    save_json(out/'source_labels.json',dict(sha256=digest(DATA/'label-100.xlsx'),
        sheet='label',notes='修复说明包含来源路径等非建模信息，原表只读保留，不在提交包复制身份或本地路径信息。'))

if __name__=='__main__':main()
