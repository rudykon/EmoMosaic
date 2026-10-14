"""Four independent single-GPU workers, fail-fast per job, resumable completed runs."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import yaml
from data import ROOT,OUT,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpus',default='0,1,2,3');ap.add_argument('--variants')
    args=ap.parse_args();cfg=yaml.safe_load((ROOT/'src/p2p3/config.yaml').read_text())
    variants=args.variants.split(',') if args.variants else cfg['variants']
    queue=[(v,s) for v in variants for s in cfg['seeds'] if not (OUT/'runs'/f'{v}_s{s}'/'complete.json').exists()]
    active={};failures=[];gpus=args.gpus.split(',')
    write_json(OUT/'runtime/experiment_plan.json',{'variants':variants,'seeds':cfg['seeds'],'gpus':gpus,
                'jobs_total':len(variants)*len(cfg['seeds']),'selection_uses':['train','valid'],
                'test_policy':'No test prediction until all training and validation selection are frozen.'})
    while queue or active:
        for gpu in gpus:
            if gpu in active:
                proc,log,variant,seed=active[gpu]
                code=proc.poll()
                if code is not None:
                    log.close();del active[gpu]
                    if code:failures.append({'variant':variant,'seed':seed,'exit_code':code})
                    print(f'finished gpu={gpu} {variant} seed={seed} exit={code}',flush=True)
            if gpu not in active and queue:
                variant,seed=queue.pop(0);run=OUT/'runs'/f'{variant}_s{seed}';run.mkdir(exist_ok=True,parents=True)
                log=open(run/'train.log','a',buffering=1)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'src/p2p3/train.py'),'--variant',variant,
                                       '--seed',str(seed),'--device','cuda:0'],env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                active[gpu]=(proc,log,variant,seed)
                print(f'start gpu={gpu} pid={proc.pid} {variant} seed={seed}',flush=True)
        write_json(OUT/'runtime/scheduler_state.json',{'pid':os.getpid(),'active':[
            {'gpu':g,'pid':p.pid,'variant':v,'seed':s} for g,(p,_,v,s) in active.items()],
            'queued':len(queue),'failures':failures,'updated':time.time()})
        if queue or active:time.sleep(5)
    write_json(OUT/'runtime/training_finished.json',{'failures':failures,'jobs':len(variants)*len(cfg['seeds'])})
    if failures:sys.exit(1)


if __name__=='__main__':main()
