"""Resume public fixed-revision safetensors in short HTTP ranges, verify LFS SHA.

Fallback for CDN streams that repeatedly interrupt hf download. Downloads only
public model files; no credentials, dataset contents or inference requests.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import shutil
import time

import requests
from huggingface_hub import HfApi

from data import ROOT
from improvement_common import EXP,digest,save_json

REPO='Qwen/Qwen2.5-7B'
REVISION='d149729398750b98c0af14eb82c78cfe92750796'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=12)
    args=ap.parse_args()
    folder=ROOT/'models/pretrained/qwen2.5-7b'
    info=HfApi().model_info(REPO,revision=REVISION,files_metadata=True)
    assert info.sha==REVISION
    files=[s for s in info.siblings if s.rfilename.endswith('.safetensors')]
    entries=[];jobs=[];verified=[]
    for entry in files:
        name=entry.rfilename;size=entry.size;expected=entry.lfs.sha256
        target=folder/name;part=folder/(name+'.part')
        if target.exists() and target.stat().st_size==size and digest(target)==expected:
            print(f'{name}: verified existing file',flush=True)
            verified.append({'name':name,'bytes':size,'sha256':expected})
            continue
        if not part.exists():
            matches=list((folder/'.cache/huggingface/download').glob(f'*.{expected}.incomplete'))
            if matches:
                max(matches,key=lambda p:p.stat().st_size).rename(part)
        ranges=folder/(name+'.ranges');ranges.mkdir(exist_ok=True)
        state=ranges/'prefix.json'
        identity={'prefix_bytes':part.stat().st_size if part.exists() else 0,'size':size,'sha256':expected}
        if state.exists() and json.loads(state.read_text())!=identity:
            raise RuntimeError('Frozen prefix changed during parallel download')
        save_json(state,identity)
        chunks=[]
        for offset in range(identity['prefix_bytes'],size,16*1024**2):
            end=min(offset+16*1024**2,size)-1
            chunk=ranges/f'{offset}_{end}.part'
            chunks.append(chunk);jobs.append((name,size,offset,end,chunk))
        entries.append((name,size,expected,target,part,chunks))
    def download(job):
        name,size,start,end,chunk=job
        session=requests.Session();failures=0
        while not chunk.exists() or chunk.stat().st_size<end-start+1:
            offset=start+(chunk.stat().st_size if chunk.exists() else 0)
            # Unique query prevents an intermediary reusing a different Range.
            url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true&part={offset}'
            try:
                with session.get(url,headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=(20,60)) as response:
                    if response.status_code!=206:
                        raise RuntimeError(f'Expected partial response, got HTTP {response.status_code}')
                    content_range=response.headers.get('Content-Range','')
                    if not content_range.startswith(f'bytes {offset}-') or not content_range.endswith(f'/{size}'):
                        raise RuntimeError('Content-Range did not match requested file offset and size')
                    written=offset
                    with chunk.open('ab') as f:
                        for block in response.iter_content(chunk_size=64*1024):
                            if block:
                                if written+len(block)>end+1:raise RuntimeError('Server exceeded requested byte range')
                                f.write(block);written+=len(block)
                    if written!=end+1:raise RuntimeError('Partial response ended early')
                failures=0
                print(f'{name}: completed range {start}-{end}',flush=True)
            except (requests.RequestException,RuntimeError) as error:
                failures+=1
                print(f'{name}: range {start}-{end} retry {failures} ({type(error).__name__})',flush=True)
                if failures>=12:raise RuntimeError(f'{name}: repeated failures; retained partial file') from None
                time.sleep(min(failures*2,20))
        if chunk.stat().st_size!=end-start+1:raise RuntimeError('Invalid saved range size')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download,jobs))
    for name,size,expected,target,part,chunks in entries:
        assembled=folder/(name+'.assembling')
        with assembled.open('wb') as stream:
            for piece in ([part] if part.exists() else [])+chunks:
                with piece.open('rb') as source:shutil.copyfileobj(source,stream,16*1024**2)
        if assembled.stat().st_size!=size:raise RuntimeError('Assembled file size mismatch')
        actual=digest(assembled)
        if actual!=expected:raise RuntimeError(f'{name}: SHA256 mismatch; partial file retained for diagnosis')
        assembled.replace(target)
        print(f'{name}: SHA256 verified',flush=True)
        verified.append({'name':name,'bytes':size,'sha256':actual})
    save_json(EXP/'qwen7_download_verified.json',{'repository':REPO,'revision':REVISION,'files':verified})


if __name__=='__main__':
    main()
