"""Resume public fixed-revision safetensors in short HTTP ranges, verify LFS SHA.

Fallback for CDN streams that repeatedly interrupt hf download. Downloads only
public model files; no credentials, dataset contents or inference requests.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import shutil
import time

import requests
from huggingface_hub import HfApi

from data import ROOT
from improvement_common import EXP,digest,save_json

from text_encoder_registry import encoder_specs


def download_specs():
    """Extra downloads do not alter the registered experiment candidates."""
    specs=encoder_specs()
    extra=EXP/'qwen35_extra_downloads.json'
    if extra.exists():
        models=json.loads(extra.read_text())['models']
        if set(models)&set(specs):
            raise ValueError('An extra download cannot replace a registered encoder')
        specs.update(models)
    # The historical 1.5B ablation may also need its exact public files restored.
    return {name:spec for name,spec in specs.items() if name.startswith('qwen35_') or name=='qwen'}


def main():
    specs=download_specs()
    ap=argparse.ArgumentParser()
    ap.add_argument('--encoder',choices=list(specs),required=True)
    ap.add_argument('--workers',type=int,default=12)
    ap.add_argument('--after-download',choices=list(specs),
                    help='Wait for this model to finish and verify before starting any network request')
    args=ap.parse_args()
    if args.after_download:
        if args.after_download==args.encoder:
            ap.error('--after-download must name a different encoder')
        prior=specs[args.after_download]
        marker=EXP/f'{args.after_download}_download_verified.json'
        last=0
        while not marker.exists():
            if time.monotonic()-last>=60:
                print(f'Waiting for verified {args.after_download} download before {args.encoder}',flush=True)
                last=time.monotonic()
            time.sleep(5)
        verified_prior=json.loads(marker.read_text())
        if (verified_prior['repository']!=prior['repository'] or
                verified_prior['revision']!=prior['revision'] or
                not any(e['name'].endswith('.safetensors') for e in verified_prior['files'])):
            raise RuntimeError('Prerequisite download identity or weights missing')
        for entry in verified_prior['files']:
            path=ROOT/prior['local_dir']/entry['name']
            if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:
                raise RuntimeError('Prerequisite download failed file verification')
        print(f'Verified {args.after_download} is complete; resuming {args.encoder}',flush=True)
    spec=specs[args.encoder]
    REPO,REVISION=spec['repository'],spec['revision']
    folder=ROOT/spec['local_dir'];folder.mkdir(parents=True,exist_ok=True)
    info=HfApi().model_info(REPO,revision=REVISION,files_metadata=True)
    assert info.sha==REVISION
    files=[s for s in info.siblings if s.rfilename.endswith('.safetensors')]
    # Configuration/tokenizer assets use the same immutable revision.
    small=[]
    for entry in info.siblings:
        name=entry.rfilename
        if name.endswith('.safetensors') or name.startswith('.'):continue
        target=folder/name
        if not target.exists() or target.stat().st_size!=entry.size:
            response=requests.get(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}',timeout=(20,120))
            response.raise_for_status()
            part=target.with_name(target.name+'.smallpart');part.write_bytes(response.content)
            if part.stat().st_size!=entry.size:raise RuntimeError('Small file size mismatch')
            part.replace(target)
        actual=digest(target)
        if entry.lfs and actual!=entry.lfs.sha256:raise RuntimeError('Small LFS file checksum mismatch')
        small.append({'name':name,'bytes':entry.size,'sha256':actual})
        print(f'{name}: downloaded and checked',flush=True)
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
    # A transient outage can exhaust a few ranges while other transfers recover.
    # Finish each round, then retry only failed ranges from their saved offsets.
    # Do not cancel healthy queued downloads because one early future failed.
    pending=jobs
    for round_index in range(3):
        failed=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(download,job):job for job in pending}
            for future in as_completed(futures):
                try:
                    future.result()
                except (requests.RequestException,RuntimeError):
                    failed.append(futures[future])
        if not failed:
            break
        if round_index==2:
            raise RuntimeError(f'{len(failed)} ranges still incomplete after recovery rounds; retained all partial data')
        print(f'Recovery round {round_index+2}: {len(failed)} incomplete ranges',flush=True)
        pending=failed
        time.sleep(20)
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
    save_json(EXP/f'{args.encoder}_download_verified.json',{'repository':REPO,'revision':REVISION,'files':verified+small})


if __name__=='__main__':
    main()
