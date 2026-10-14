"""Sequential exploration phases; a filesystem lock prevents duplicate queues.

Use from the project root after prepare.py has built the aligned input caches.
No test metrics are computed until select_improvement.py has frozen all choices.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

from project_paths import DATA_ROOT
from project_paths import python_executable
from data import ROOT
from text_encoder_registry import active_encoders,encoder_specs
from improvement_common import EXP,digest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['small','bert_svd','qwen7','qwen35_08','qwen35_2','qwen35_4','qwen35','evaluate','deliver','all'],required=True)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--wait-download',action='store_true',help='Wait for an already running downloader; do not launch another')
    ap.add_argument('--wait-experiments',action='store_true',help='Wait for already running experiment branches before the evaluation phase')
    ap.add_argument('--wait-evaluation',action='store_true',help='Wait for the frozen evaluation, raw-interface check and complete attachment3 CSV before delivery')
    args=ap.parse_args();EXP.mkdir(parents=True,exist_ok=True)
    (EXP/'runtime').mkdir(parents=True,exist_ok=True)
    lock=(EXP/'runtime'/f'pipeline_{args.phase}.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads))
    def run(name,*argv):
        python=sys.executable
        if name in ['extract_text_experts.py','verify_qwen35_model.py'] and any(str(a).startswith('qwen35_') for a in argv):
            python=python_executable('qwen35')
        command=[python,str(ROOT/'src/p2p3'/name),*map(str,argv)]
        print('Running: '+' '.join(command),flush=True)
        subprocess.run(command,cwd=ROOT,env=env,check=True)
    def encoder(name):
        if name.startswith('qwen35_') and not (EXP/f'{name}_model_verification.json').exists():
            run('verify_qwen35_model.py','--encoder',name,'--threads',args.threads)
        if name=='qwen35_08' and not (EXP/'pipeline_smoke'/name/'complete.json').exists():
            run('smoke_qwen35_pipeline.py','--encoder',name,'--threads',args.threads)
        if name!='bert_svd':run('extract_text_experts.py','--encoder',name,'--splits','train,valid,attachment3,attachment4','--threads',args.threads)
        run('probe_text_experts.py','--encoder',name)
        run('cross_validate_text_expert.py','--encoder',name)
        run('explore_text_fusion.py','--encoder',name)
        selected=json.loads((EXP/'probes'/name/'selection.json').read_text())
        if name.startswith('bert') or selected['fusion_gate_passed']:
            run('strict_text_intervention.py','--encoder',name,'--threads',args.threads)
    if args.phase in ['small','all']:
        run('explore_existing.py')
        encoder('bert');encoder('bert_svd');encoder('qwen')
        run('run_improvement_training.py','--workers',2,'--threads',max(1,args.threads//2))
        run('run_improvement_training.py','--layer-mix','--weights','0.3','--workers',1,'--threads',args.threads)
        run('lora_text_expert.py','--threads',args.threads)
        run('cross_validate_lora.py','--threads',args.threads)
        run('verify_lora_reload.py','--threads',args.threads)
    if args.phase=='bert_svd':encoder('bert_svd')
    if args.phase=='qwen7':
        marker=EXP/'qwen7_download_verified.json'
        if args.wait_download:
            deadline=time.monotonic()+6*3600;last=0
            while not marker.exists():
                if time.monotonic()>deadline:raise TimeoutError('Public weights still incomplete; inspect downloader before resuming')
                if time.monotonic()-last>60:
                    print('Waiting for verified public Qwen7 weights',flush=True);last=time.monotonic()
                time.sleep(5)
        elif not marker.exists():run('download_qwen7_resumable.py')
        manifest=json.loads(marker.read_text())
        for entry in manifest['files']:
            if digest(ROOT/'models/pretrained/qwen2.5-7b'/entry['name'])!=entry['sha256']:
                raise RuntimeError('Public Qwen7 weights failed checksum')
        encoder('qwen7')
    if args.phase in ['qwen35_08','qwen35_2','qwen35_4','qwen35','all']:
        names=['qwen35_08','qwen35_2','qwen35_4'] if args.phase in ['qwen35','all'] else [args.phase]
        for name in names:
            marker=EXP/f'{name}_download_verified.json'
            if args.wait_download:
                deadline=time.monotonic()+6*3600;last=0
                while not marker.exists():
                    if time.monotonic()>deadline:raise TimeoutError(f'{name} weights incomplete')
                    if time.monotonic()-last>60:
                        print(f'Waiting for verified public {name} weights',flush=True);last=time.monotonic()
                    time.sleep(5)
            elif not marker.exists():run('download_qwen35_resumable.py','--encoder',name)
            manifest=json.loads(marker.read_text())
            for entry in manifest['files']:
                if digest(ROOT/encoder_specs()[name]['local_dir']/entry['name'])!=entry['sha256']:
                    raise RuntimeError(f'{name} public file checksum failed')
            encoder(name)
    if args.phase in ['evaluate','all']:
        if args.wait_experiments:
            from select_improvement import readiness
            from improvement_common import initialize_protocol
            deadline=time.monotonic()+6*3600;last=0
            while True:
                try:
                    readiness(initialize_protocol())
                    break
                except RuntimeError:
                    missing=json.loads((EXP/'freeze_readiness.json').read_text())['missing']
                    if not missing:raise
                    if time.monotonic()>deadline:raise TimeoutError('Experiments still incomplete; inspect their live handles')
                    if time.monotonic()-last>60:
                        print(f'Waiting for {len(missing)} experiment prerequisites',flush=True);last=time.monotonic()
                    time.sleep(5)
        if not (EXP/'final_selection/frozen.json').exists():
            # Verify the all-branch validation winner's public interface before
            # freezing inference source hashes and opening any test labels.
            run('select_improvement.py','--preview-completed','--threads',args.threads)
            run('verify_improved_interface.py','--config',EXP/'completed_branch_preview/decision_rules.json',
                '--output',EXP/'completed_branch_preview/pre_freeze_interface.json')
        run('select_improvement.py','--threads',args.threads)
        # A missing public checkpoint must not block preparation of the other
        # frozen feature caches. --wait-download waits for an existing restore;
        # model identities and cache hashes are still checked by extraction and
        # evaluation before any test metrics are computed.
        pending=active_encoders();deadline=time.monotonic()+6*3600;last=0
        while pending:
            for name in list(pending):
                manifest=json.loads((EXP/'features'/name/'encoder.json').read_text())
                weights=manifest['weights_sha256']
                names=['model.safetensors'] if isinstance(weights,str) else list(weights)
                folder=ROOT/encoder_specs()[name]['local_dir']
                if not all((folder/file).is_file() for file in ['config.json','tokenizer_config.json',*names]):
                    if not args.wait_download:
                        raise FileNotFoundError(f'{name} public files missing at {folder}; restore the pinned revision or wait for its downloader with --wait-download')
                    continue
                run('extract_text_experts.py','--encoder',name,'--splits','test','--threads',args.threads)
                pending.remove(name)
            if pending:
                if time.monotonic()>deadline:raise TimeoutError(f'Public files still missing for frozen test extraction: {pending}')
                if time.monotonic()-last>60:
                    print(f'Waiting for public files for frozen test extraction: {pending}',flush=True);last=time.monotonic()
                time.sleep(5)
        run('evaluate_improvement.py','--threads',args.threads)
        config=EXP/'final_selection/decision_rules.json'
        run('strict_selected_validation.py','--config',config,'--threads',args.threads)
        run('verify_improved_interface.py','--config',config,'--output',EXP/'final_interface_verification.json')
        run('predict.py','--improvement-config',config,'--input-dir',DATA_ROOT / 'source_data/attachment3_missing_modalities/aligned',
            '--output',EXP/'attachment3_predictions.csv')
    if args.phase in ['deliver','all']:
        if args.wait_evaluation:
            import csv
            deadline=time.monotonic()+8*3600;last=0
            required=[EXP/'final_selection/frozen.json',EXP/'final_selection/decision_rules.json',
                      EXP/'final_evaluation/complete.json',EXP/'final_interface_verification.json',EXP/'attachment3_predictions.csv']
            while True:
                ready=all(p.is_file() for p in required)
                if ready:
                    frozen=json.loads(required[0].read_text())
                    config_sha=digest(required[1])
                    evaluation=json.loads(required[2].read_text())
                    interface=json.loads(required[3].read_text())
                    strict=[json.loads(p.read_text()) for p in (EXP/'strict_selected').glob('*/complete.json')]
                    with required[4].open(encoding='utf-8-sig',newline='') as stream:
                        rows=list(csv.DictReader(stream))
                    ready=(config_sha==frozen['decision_sha256']
                           and evaluation['frozen_sha256']==digest(required[0])
                           and interface['config_sha256']==config_sha
                           and len(rows)==30 and all(row.get('attribution_target')=='RM-Net core intensity' for row in rows)
                           and any(r['identity']['config_sha256']==config_sha for r in strict))
                if ready:break
                if time.monotonic()>deadline:raise TimeoutError('Evaluation deliverables incomplete; inspect their live process before resuming')
                if time.monotonic()-last>60:
                    print('Waiting for frozen evaluation, final strict/interface verification and all 30 CSV rows',flush=True)
                    last=time.monotonic()
                time.sleep(5)
    print(f'Completed phase: {args.phase}',flush=True)


if __name__=='__main__':main()
