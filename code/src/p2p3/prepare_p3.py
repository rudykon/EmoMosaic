"""Rebuild caches with frozen P2 statistics, without refitting or touching P2."""
import argparse
import os
from pathlib import Path
import json
import numpy as np
import torch
from transformers import BertModel
from project_paths import DATA_ROOT
from data import ROOT,MODALITIES,load_pickle,unwrap,raw_arrays,normalize,write_json
from predict import Predictor
from p3_common import OUT3,attachment_paths,digest,provenance


def main():
    global OUT3
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--force',action='store_true')
    ap.add_argument('--output-root', type=Path, help='New P3 cache directory; defaults to outputs/reproduced/prepared_p3')
    args=ap.parse_args()
    OUT3 = (args.output_root or Path(os.environ.get('COLING_P3_OUTPUT_ROOT', ROOT/'outputs/reproduced/prepared_p3'))).resolve()
    if OUT3 == ROOT/'outputs/problem3':
        ap.error('Use an independent output-root to preserve historical results')
    torch.set_num_threads(2);p=Predictor(device=args.device,load_models=False)
    out=OUT3/'cache';out.mkdir(parents=True,exist_ok=True)
    source=DATA_ROOT/'source_data/attachment2_features/aligned_50.pkl';raw=load_pickle(source)
    bert=None;report=dict(source_sha256=digest(source),assets=provenance(),splits={})
    for split in ['valid','test']:
        path=out/f'{split}.npz'
        if path.exists() and not args.force:continue
        # Prefer exact already prepared P2 arrays when available, with copied
        # bytes and source digest. Rebuild only in an unpacked clean workspace.
        previous=ROOT/'outputs/problem2/cache'/f'{split}.npz'
        if previous.exists() and not args.force:
            import shutil
            shutil.copyfile(previous,path);report['splits'][split]=dict(mode='exact_P2_cache_copy',sha256=digest(previous));continue
        if bert is None:bert=BertModel.from_pretrained(ROOT/'models/pretrained/bert',local_files_only=True).to(p.device).eval()
        d=raw[split];tb=np.asarray(d['text_bert']);text=[]
        for start in range(0,len(tb),64):
            b=torch.as_tensor(tb[start:start+64],device=p.device,dtype=torch.long)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=p.device.type=='cuda'):
                h=bert(input_ids=b[:,0],attention_mask=b[:,1],token_type_ids=b[:,2]).last_hidden_state
            text.append(h.float().cpu().numpy()*tb[start:start+64,1,:,None])
        text=np.concatenate(text);arrays,obs,lengths=raw_arrays(d,text);xx=normalize(arrays,obs,p.stats)
        np.savez(path,**dict(zip(MODALITIES,xx)),obs=obs,lengths=lengths,
            y=np.asarray(d['regression_labels'],np.float32).reshape(-1),ids=np.asarray(d['id'],str),
            raw_text=np.asarray(d['raw_text'],str),token_ids=tb[:,0].astype(np.int64))
        valid=np.arange(50)[None]<lengths[:,None]
        report['splits'][split]=dict(mode='fixed_BERT_reencoded',official_text_mae=float(abs(text[valid]-np.asarray(d['text'])[valid]).mean()),sha256=digest(path))
    path=out/'attachment4.npz'
    if not path.exists() or args.force:
        arrays={m:[] for m in MODALITIES};oo=[];ll=[];ids=[];toks=[];files=[]
        for f in attachment_paths():
            d=p.load_file(f);t=p.input_tensors(d)
            for j,m in enumerate(MODALITIES):arrays[m].append(t[0][j].cpu().numpy())
            oo.append(t[1].cpu().numpy());ll.append(t[2].cpu().numpy());ids.append(str(d['id']))
            toks.append(np.asarray(d['text_bert'])[0]);files.append(f.name)
        np.savez(path,**{m:np.concatenate(v) for m,v in arrays.items()},obs=np.concatenate(oo),lengths=np.concatenate(ll),
            ids=np.array(ids),files=np.array(files),token_ids=np.asarray(toks,dtype=np.int64))
    write_json(OUT3/'preparation.json',report)
    print('P3 cache ready; training statistics and model unchanged.')


if __name__=='__main__':main()
