"""Frozen SVD/logistic vs RM-Net evaluation; writes only to a new audit folder.

Run with the project .venv-p2 Python and four BLAS threads. The existing missingness
grid is evaluated on test, with duplicate effective masks computed once. No model
or decision rule is chosen using the resulting test scores.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import sklearn
import torch
from scipy.special import softmax

import boosted_classifier as boosted
from data import ROOT, OUT, MODALITIES, load_split
from evaluate import scenarios, scenario_mask
from metrics import classes, classification
from predict import Predictor
from train import device_data, validation_masks

MODEL_NAMES = ('svd_lr', 'rmnet_formal', 'rmnet_aux_probability',
               'rmnet_aux_logits', 'svd_lr_before_bias')
STATE = {}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_mask(data, row, replay_records):
    if row['family'] == 'clean':
        return np.ones_like(data['obs'])
    if row['family'] == 'mixed':
        return validation_masks(data['lengths'], row['seed'])[
            [.2, .4, .6].index(row['rate'])]
    if row['family'] == 'replay':
        record = replay_records[row['pattern_index']]
        keep = np.ones_like(data['obs'])
        reference = record['valid_len']
        for i, length in enumerate(data['lengths']):
            for m, name in enumerate(MODALITIES):
                for start, end in record[f'miss_{name}_intervals']:
                    lo = int(np.floor(start / reference * length))
                    hi = min(length, int(np.ceil(end / reference * length)))
                    keep[i, m, lo:hi] = False
        return keep
    return scenario_mask(data['lengths'], row)


def prepare_masked(data, observed):
    result = dict(data, obs=observed)
    for j, name in enumerate(MODALITIES):
        # The frozen classifier's endpoints assume already-zeroed missing input.
        result[name] = np.where(observed[:, j, :, None], data[name], 0)
    return result


@torch.inference_mode()
def rm_predict(predictor, tensors, observed, batch=96):
    xx, oo, ll = tensors
    pieces = []
    for start in range(0, len(ll), batch):
        sl = slice(start, start + batch)
        mask = torch.as_tensor(observed[sl], dtype=torch.bool)
        parts = [m([x[sl] for x in xx], oo[sl], ll[sl], mask)
                 for m in predictor.models]
        y = torch.stack([p['y'].float() for p in parts]).mean(0)
        logits = torch.stack([p['logits'].float() for p in parts]).mean(0)
        probs = torch.stack([p['logits'].float().softmax(-1) for p in parts]).mean(0)
        pieces.append((y.numpy(), logits.numpy(), probs.numpy()))
    return {name: np.concatenate([p[i] for p in pieces])
            for i, name in enumerate(('intensity', 'logits', 'aux_probs'))}


def worker_init(model_path, threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    STATE.update(model=joblib.load(model_path), predictor=Predictor(device='cpu'),
                 datasets={}, tensors={})


def worker_predict(task):
    split, key, mask_path, pred_path = task
    if split not in STATE['datasets']:
        STATE['datasets'][split] = load_split(split)
        STATE['tensors'][split] = device_data(STATE['datasets'][split], 'cpu')
    data = STATE['datasets'][split]
    with np.load(mask_path) as z:
        observed = z['observed']
    start = time.perf_counter()
    new = boosted.predict(STATE['model'], prepare_masked(data, observed))
    new_seconds = time.perf_counter() - start
    start = time.perf_counter()
    old = rm_predict(STATE['predictor'], STATE['tensors'][split], observed)
    old_seconds = time.perf_counter() - start
    raw = new['scores'].copy()
    raw[:, 0] -= STATE['model']['bias'][0]
    raw[:, 2] -= STATE['model']['bias'][1]
    prediction = np.stack([
        new['class_id'], classes(old['intensity'], STATE['predictor'].thresholds),
        old['aux_probs'].argmax(1), old['logits'].argmax(1), raw.argmax(1)
    ]).astype(np.int8)
    for values in (new['probabilities'], old['aux_probs']):
        assert np.isfinite(values).all()
        assert np.allclose(values.sum(1), 1., atol=1e-6)
    assert np.isfinite(old['intensity']).all()
    np.savez_compressed(pred_path, predictions=prediction, ids=data['ids'],
                        truth=boosted._labels(data['y']), intensity=old['intensity'],
                        svd_scores=new['scores'], svd_probabilities=new['probabilities'],
                        auxiliary_probabilities=old['aux_probs'],
                        auxiliary_logits=old['logits'])
    return dict(split=split, key=key, svd_seconds=new_seconds, rmnet_seconds=old_seconds)


def audit_inputs(source_run, output):
    original = json.loads((source_run / 'run_manifest.json').read_text())
    ids, videos, audits = {}, {}, {}
    for split in ('train', 'valid', 'test'):
        data = load_split(split)
        ids[split] = set(map(str, data['ids']))
        videos[split] = {i.split('$_$')[0] for i in ids[split]}
        assert len(ids[split]) == len(data['ids'])
        assert all(np.isfinite(data[m]).all() for m in MODALITIES)
        assert all(not np.any(data[m][~data['obs'][:, j]])
                   for j, m in enumerate(MODALITIES))
        cache_hash = digest(OUT / 'cache' / f'{split}.npz')
        assert cache_hash == original['cache_sha256'][split]
        audits[split] = dict(n=len(data['ids']), videos=len(videos[split]),
                            class_counts=np.bincount(boosted._labels(data['y']), minlength=3).tolist(),
                            cache_sha256=cache_hash)
        del data
        gc.collect()
    for a, b in (('train', 'valid'), ('train', 'test'), ('valid', 'test')):
        assert not ids[a] & ids[b]
        assert not videos[a] & videos[b]
    hashes = {path: digest(ROOT / path) for path in original['preserved_rmnet_sha256']}
    assert hashes == original['preserved_rmnet_sha256']
    source_hash = digest(ROOT / 'src/p2p3/boosted_classifier.py')
    assert source_hash == original['source_sha256']
    model_hash = digest(source_run / 'model.joblib')
    assert model_hash == json.loads((source_run / 'evaluation.json').read_text())['model_sha256']
    audits.update(no_sample_overlap=True, no_source_video_overlap=True,
                  rmnet_assets_match_previous_run=True, frozen_model_sha256=model_hash,
                  app_model_matches_frozen=digest(ROOT / 'app_demo/model.joblib') == model_hash,
                  source_sha256=source_hash, protected_assets=hashes)
    write_json(output / 'input_audit.json', audits)
    return audits


def reproduce_training(source_run, output):
    start = time.perf_counter()
    train, valid = load_split('train'), load_split('valid')
    result = boosted.fit(train, valid, output / 'reproduced_model.joblib')
    del train
    gc.collect()
    fitted = joblib.load(output / 'reproduced_model.joblib')
    frozen = joblib.load(source_run / 'model.joblib')
    comparison = {}
    for split, data in (('valid', valid), ('test', load_split('test'))):
        old = boosted.predict(frozen, data)
        new = boosted.predict(fitted, data)
        comparison[split] = dict(class_disagreements=int(np.sum(old['class_id'] != new['class_id'])),
                                max_score_difference=float(np.max(abs(old['scores'] - new['scores']))))
        assert comparison[split]['class_disagreements'] == 0
    result.update(reproduction_comparison=comparison, elapsed_seconds=time.perf_counter()-start,
                  classifier_iterations=fitted['classifier'].n_iter_.tolist(),
                  test_used_for_training_or_selection=False)
    write_json(output / 'training_reproduction.json', result)
    del valid, fitted, frozen
    gc.collect()


def build_protocol(output):
    replay = json.loads((OUT / 'data_audit.json').read_text())['special']['attachment3']
    rows = []
    for split in ('valid', 'test'):
        rows.extend([dict(split=split, family='clean', kind='TAV', rate=0., seed=201),
                     dict(split=split, family='mixed', kind='mixed', rate=.4, seed=20260923)])
    # Preserve all previously specified conditions, including duplicate placements
    # at deterministic positions and full disappearance. Deduplicate computation.
    rows.extend(dict(row, split='test') for row in scenarios() if row['family'] != 'clean')
    for rate in (.2, .4, .6):
        for seed in range(20260923, 20260928):
            if rate == .4 and seed == 20260923:
                continue
            rows.append(dict(split='test', family='mixed', kind='mixed', rate=rate, seed=seed))
    rows.extend(dict(split='test', family='replay', kind='attachment3',
                     pattern_index=i, file=record['file']) for i, record in enumerate(replay))
    write_json(output / 'protocol.json', dict(
        description='Frozen evaluation plan saved before this expanded inference run; previous headline results were already known.',
        models=list(MODEL_NAMES), class_order=boosted.CLASS_NAMES.tolist(),
        selection='No test-based parameter, threshold, feature, or model selection.',
        primary_comparisons='SVD minus formal and probability-mean auxiliary on clean and original 40% mixed test.',
        uncertainty='10000 paired source-video cluster bootstrap resamples; 20000 cluster model-swap permutations for four primary F1 contrasts, Holm-adjusted.',
        secondary='Logit-mean auxiliary and unbiased SVD; grid, position, duration, fragmentation, colocation and replay are exploratory.',
        precision='All fresh inference CPU float32; frozen RM-Net FP16 checkpoints loaded into float32 modules; class decisions checked against stored original outputs.',
        sample_dependence='Resample whole source videos, retaining all clips and all repeated masks.',
        existing_protocol_sha256=digest(ROOT / 'models/rmnet/eval_missing_configs.json'),
        scenarios=rows))
    tasks, unique = [], {}
    data = {split: load_split(split) for split in ('valid', 'test')}
    for index, row in enumerate(rows):
        split = row['split']
        observed = data[split]['obs'] & make_mask(data[split], row, replay)
        key = hashlib.sha256(observed.tobytes()).hexdigest()
        row.update(audit_scenario_id=index, mask_key=key)
        if (split, key) not in unique:
            mask_path = output / 'masks' / f'{split}_{key}.npz'
            pred_path = output / 'predictions' / f'{split}_{key}.npz'
            np.savez_compressed(mask_path, observed=observed)
            task = (split, key, str(mask_path), str(pred_path))
            unique[split, key] = task
            tasks.append(task)
    write_json(output / 'scenarios.json', rows)
    del data
    gc.collect()
    return rows, tasks


def summarize(output, rows):
    summaries, primary_checks = [], []
    baseline = json.loads((OUT / 'ensemble_metrics.json').read_text())
    for row in rows:
        with np.load(output / 'predictions' / f"{row['split']}_{row['mask_key']}.npz") as z:
            truth = z['truth']
            for j, name in enumerate(MODEL_NAMES):
                measured = classification(truth - 1, z['predictions'][j])
                measured.pop('confusion')
                summaries.append(dict(row, model=name, n=len(truth), **measured))
            if row['audit_scenario_id'] < 4:
                condition = 'clean' if row['family'] == 'clean' else 'missing'
                with np.load(OUT / f"{row['split']}_ensemble_{condition}.npz") as old:
                    rules = json.loads((ROOT/'models/rmnet/decision_rules.json').read_text())
                    check = dict(split=row['split'], condition=condition,
                        formal_class_disagreements=int(np.sum(classes(old['y'], rules['thresholds']) != z['predictions'][1])),
                        auxiliary_class_disagreements=int(np.sum(old['auxiliary_probs'].argmax(1) != z['predictions'][2])),
                        max_intensity_difference=float(np.max(abs(old['y'] - z['intensity']))),
                        max_aux_probability_difference=float(np.max(abs(old['auxiliary_probs'] - z['auxiliary_probabilities']))))
                    assert check['formal_class_disagreements'] == 0, check
                    assert check['auxiliary_class_disagreements'] == 0, check
                    assert np.isclose(classification(truth-1, z['predictions'][1])['macro_f1'],
                                      baseline[row['split']][condition]['macro_f1'])
                    primary_checks.append(check)
                output_rows = []
                for i, sample_id in enumerate(z['ids']):
                    output_rows.append(dict(sample_id=str(sample_id), video_id=str(sample_id).split('$_$')[0],
                        true_class=int(truth[i]), **{name: int(z['predictions'][j,i]) for j,name in enumerate(MODEL_NAMES)}))
                write_csv(output / f"{row['split']}_{condition}_predictions.csv", output_rows)
    write_csv(output / 'all_metrics.csv', summaries)
    write_json(output / 'baseline_reproduction.json', primary_checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, default=OUT/'boosted_classifier/20260925_225440')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    for folder in ('masks','predictions'):
        (output/folder).mkdir(exist_ok=True)
    start=time.perf_counter()
    torch.set_num_threads(args.threads)
    if args.resume:
        audit = json.loads((output/'input_audit.json').read_text())
        rows = json.loads((output/'scenarios.json').read_text())
        unique = {(r['split'],r['mask_key']) for r in rows}
        tasks=[(s,k,str(output/'masks'/f'{s}_{k}.npz'),str(output/'predictions'/f'{s}_{k}.npz')) for s,k in sorted(unique)]
    else:
        audit=audit_inputs(args.source_run,output)
        print('Input hashes, finite features, zeroed masks and video-disjoint splits verified.',flush=True)
        reproduce_training(args.source_run,output)
        print('Training reproduced with unchanged validation/test class predictions.',flush=True)
        rows,tasks=build_protocol(output)
        write_json(output/'environment.json',dict(python=sys.version,numpy=np.__version__,
            sklearn=sklearn.__version__,torch=torch.__version__,platform=platform.platform(),
            cuda_available=torch.cuda.is_available(),workers=args.workers,threads_per_worker=args.threads,
            command=sys.argv,source_sha256=digest(__file__)))
    pending=[t for t in tasks if not Path(t[3]).exists()]
    print(f'{len(rows)} declared scenarios; {len(tasks)} distinct masks; {len(pending)} to infer.',flush=True)
    timings=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=worker_init,
            initargs=(str(args.source_run/'model.joblib'),args.threads)) as pool:
        futures={pool.submit(worker_predict,t):t for t in pending}
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            timings.append(future.result())
            if i%10==0 or i==len(pending):
                progress=dict(completed=i,total=len(pending),elapsed_seconds=time.perf_counter()-start)
                write_json(output/'progress.json',progress)
                print(json.dumps(progress),flush=True)
    write_csv(output/'inference_timings.csv',timings)
    summarize(output,rows)
    assert digest(args.source_run/'model.joblib')==audit['frozen_model_sha256']
    assert all(digest(ROOT/p)==h for p,h in audit['protected_assets'].items())
    write_json(output/'inference_complete.json',dict(scenarios=len(rows),distinct_masks=len(tasks),
        frozen_assets_unchanged=True,elapsed_seconds=time.perf_counter()-start))
    print('All inference and baseline reproduction checks complete.',flush=True)


if __name__=='__main__':
    main()
