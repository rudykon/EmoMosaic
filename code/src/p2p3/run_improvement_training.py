"""Bounded local process queue for the 3 CE weights x 15 seeds exploration."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
import time

from data import ROOT
from improvement_common import EXP, initialize_protocol, save_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--weights',default='0.3,1,2')
    ap.add_argument('--exclude',default='',help='Explicit currently running run names; never restart them')
    ap.add_argument('--layer-mix',action='store_true')
    args=ap.parse_args()
    protocol=initialize_protocol()
    seeds=protocol['training_seeds'][:5] if args.layer_mix else protocol['training_seeds']
    tasks=[]
    excluded=set(args.exclude.split(','))
    prefix='mix_' if args.layer_mix else ''
    for weight in [float(x) for x in args.weights.split(',')]:
        for seed in seeds:
            name=f'{prefix}ce{weight:g}_s{seed}'
            run=EXP/'runs'/name
            if name in excluded or (run/'complete.json').exists():
                continue
            if (run/'history.json').exists():
                raise RuntimeError(f'Existing incomplete run {name}: verify its process before any restart')
            tasks.append((name,weight,seed,run))
    logs=EXP/'training_logs'
    logs.mkdir(parents=True,exist_ok=True)
    started=time.time()
    def execute(task):
        name,weight,seed,run=task
        cmd=[sys.executable,str(ROOT/'src/p2p3/train.py'),'--variant','A0','--seed',str(seed),
             '--device','cpu','--lambda-class',str(weight),'--selection','auxiliary_f1','--threads',str(args.threads),
             '--run-dir',str(run)]
        if args.layer_mix:
            cmd+=['--text-layers',str(EXP/'features/bert')]
        env=dict(os.environ)
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
            env[key]=str(args.threads)
        print(f'Start {name}',flush=True)
        with (logs/f'{name}.log').open('w') as log:
            p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
        result={'name':name,'returncode':p.returncode,'complete':(run/'complete.json').exists()}
        print(json.dumps(result),flush=True)
        return result
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs=[executor.submit(execute,t) for t in tasks]
        for future in as_completed(jobs):
            results.append(future.result())
            save_json(logs/f'{prefix}queue_progress.json',{'expected':len(tasks),'finished':results,'seconds':time.time()-started})
    if any(r['returncode'] or not r['complete'] for r in results):
        raise RuntimeError('Training failures: inspect per-run logs')
    save_json(logs/f'{prefix}queue_complete.json',{'finished':results,'seconds':time.time()-started})


if __name__=='__main__':
    main()
