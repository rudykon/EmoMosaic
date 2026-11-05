"""Run the frozen evaluation protocol across available GPUs."""
import argparse
import os
import subprocess
import sys
import time
import yaml
from data import ROOT,OUT,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpus',default='0,1,2,3');args=ap.parse_args()
    cfg=yaml.safe_load((ROOT/'src/p2p3/config.yaml').read_text())
    if not (OUT/'selection.json').exists():raise RuntimeError('Run freeze_selection.py first')
    queue=[]
    for split in ['valid','test']:
        for v in cfg['variants']:
            for seed in cfg['seeds']:
                run=OUT/'runs'/f'{v}_s{seed}'
                if not (OUT/'evaluation'/run.name/f'{split}_complete.json').exists():queue.append((run,split))
    active={};failures=[];gpus=args.gpus.split(',')
    while queue or active:
        for gpu in gpus:
            if gpu in active:
                proc,log,run,split=active[gpu];code=proc.poll()
                if code is not None:
                    log.close();del active[gpu]
                    if code:failures.append({'run':run.name,'split':split,'code':code})
                    print('finished',run.name,split,code,flush=True)
            if gpu not in active and queue:
                run,split=queue.pop(0);folder=OUT/'evaluation'/run.name;folder.mkdir(parents=True,exist_ok=True)
                log=(folder/f'{split}.log').open('a')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'src/p2p3/evaluate.py'),'--run',str(run),
                                       '--split',split,'--device','cuda:0'],env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                active[gpu]=(proc,log,run,split);print('started',gpu,proc.pid,run.name,split,flush=True)
        write_json(OUT/'runtime/evaluation_state.json',{'pid':os.getpid(),'active':[
            {'gpu':g,'pid':p.pid,'run':r.name,'split':s} for g,(p,_,r,s) in active.items()],
            'queued':len(queue),'failures':failures,'updated':time.time()})
        if queue or active:time.sleep(5)
    write_json(OUT/'runtime/evaluation_finished.json',{'failures':failures})
    if failures:sys.exit(1)


if __name__=='__main__':main()
