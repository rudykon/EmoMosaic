"""Reevaluate the frozen COLING predictor and RM core in a fresh output directory.

Only bundled data and the deployed BERT/RM/LoRA assets are used. No training,
configuration selection, Qwen models, or historical search caches are required.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

from data import ROOT, load_split, write_json
from improvement_common import digest, metrics
from missing import TYPES, fixed_mask
from train import validation_masks
from analyze_boosted_comparison import grouped_confusion, paired_comparison, holm
from evaluate_improvement import paired_mean_views
from p3_fusion import CONFIG, FusionExplainer


def verify_assets():
    """Verify frozen weights/config and pinned public files before loading data."""
    model_dir = CONFIG.parent
    manifest = json.loads((model_dir/'asset_manifest.json').read_text())
    if digest(CONFIG) != manifest['config_sha256']:
        raise RuntimeError('Deployed decision rules do not match asset_manifest.json')
    checked = {str(CONFIG.relative_to(ROOT)): digest(CONFIG)}
    for name, record in manifest['assets'].items():
        actual = digest(ROOT/name)
        if actual != record['sha256']:
            raise RuntimeError(f'Deployed asset checksum mismatch: {name}')
        checked[name] = actual
    entries = json.loads((ROOT/'models/pretrained/manifest.json').read_text())
    for entry in entries:
        if not entry['path'].startswith('bert/'):
            continue
        path = ROOT/'models/pretrained'/entry['path']
        actual = digest(path)
        if actual != entry['sha256']:
            raise RuntimeError(f'Pinned BERT file checksum mismatch: {entry["path"]}')
        checked[str(path.relative_to(ROOT))] = actual
    stats = ROOT/'models/rmnet/norm_stats.json'
    # Pin the original train-only statistics independently of experiment outputs.
    actual = digest(stats)
    if actual != '9fa35dfc8affdab5d2d46afec66d7af11482dd81bad1aa47d06f373646034edb':
        raise RuntimeError('Frozen train-only normalization checksum mismatch')
    checked[str(stats.relative_to(ROOT))] = actual
    return checked


def conditions(data, selected):
    if 'clean' in selected:
        yield 'clean', 'clean', None
    if 'mixed' in selected:
        yield 'mixed40', 'mixed', validation_masks(data['lengths'], 20260923)[1]
    if 'grid' in selected:
        for kind in TYPES:
            for rate in (.2, .4, .6):
                yield f'{kind}_{rate:.1f}', 'rate', fixed_mask(data['lengths'], kind, rate, 20260923)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', default='valid,test')
    parser.add_argument('--conditions', default='clean,mixed,grid',
                        help='Comma-separated clean,mixed,grid; grid is the 21-condition protocol')
    parser.add_argument('--bootstrap', type=int, default=5000,
                        help='Video-cluster bootstrap replicates for classification comparisons; 0 disables')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path,
                        help='Fresh output directory; defaults to outputs/reproduced/frozen_evaluation_<UTC>')
    parser.add_argument('--limit', type=int, help='Smoke check only: evaluate first N samples per split')
    parser.add_argument('--check-assets', action='store_true', help='Verify bundled frozen assets and exit')
    args = parser.parse_args()
    splits = args.splits.split(',')
    selected = args.conditions.split(',')
    if not splits or any(s not in ('valid', 'test') for s in splits):
        parser.error('--splits must contain valid and/or test')
    if not selected or any(s not in ('clean', 'mixed', 'grid') for s in selected):
        parser.error('--conditions must contain clean, mixed, and/or grid')
    if args.bootstrap < 0 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error('--bootstrap must be nonnegative; --threads and --limit must be positive')
    assets = verify_assets()
    if args.check_assets:
        print(json.dumps(dict(passed=True, verified_assets=len(assets), sha256=assets), indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output or ROOT/'outputs/reproduced'/f'frozen_evaluation_{stamp}').resolve()
    protected = ROOT/'outputs/problem2'
    protected_p3 = ROOT/'outputs/problem3'
    if any(output == p or p in output.parents for p in (protected.resolve(), protected_p3.resolve())):
        parser.error('Choose a separate output directory to preserve historical experiment results')
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; choose a fresh --output')
    output.mkdir(parents=True, exist_ok=True)
    (output/'predictions').mkdir()
    torch.set_num_threads(args.threads)
    # The explanation engine exposes exact final probabilities and the underlying
    # auxiliary-softmax RM baseline; its hidden caches stay inside this new run.
    import p3_fusion
    p3_fusion.OUT3 = output
    predictor = FusionExplainer(args.device)
    protocol = dict(schema='coling_frozen_reevaluation_v1', asset_sha256=assets,
                    splits=splits, conditions=selected, mask_seed=20260923,
                    sample_limit=args.limit, sample_limited=args.limit is not None,
                    bootstrap_replicates=args.bootstrap, bootstrap_seed=20260926,
                    resampling_unit='original_video', selection_performed=False,
                    historical_test_exposure=True,
                    baseline='Mean auxiliary-head softmax probabilities of five frozen RM models; mean RM intensity',
                    final='Frozen deployed RM+BERT LoRA availability-gated fusion and fixed class bias',
                    regression_intervals='not computed by this entry point',
                    classification_inference='Paired video-cluster bootstrap and model-swap permutation; clip-level exact McNemar is descriptive',
                    two_view_aggregation='Arithmetic mean of clean and mixed40 metrics within each shared video resample', input_sha256={})
    write_json(output/'protocol.json', protocol)
    rows, statistics = [], []
    for split in splits:
        data = load_split(split)
        protocol['input_sha256'][split] = digest(ROOT/f'outputs/problem2/cache/{split}.npz')
        if args.limit is not None:
            data = {key: value[:args.limit] for key, value in data.items()}
        predictor.prepare(data, split)
        indices = np.arange(len(data['ids']))
        labels = np.sign(data['y']).astype(int)+1
        _, group_index = np.unique([str(s).split('$_$')[0] for s in data['ids']], return_inverse=True)
        n_groups = int(group_index.max()+1)
        rng = np.random.default_rng(20260926)
        weights = rng.multinomial(n_groups, np.full(n_groups, 1/n_groups), size=args.bootstrap) if args.bootstrap else None
        signs = rng.choice([-1,1], size=(args.bootstrap,n_groups)) if args.bootstrap else None
        headline_cms = {}
        for scenario, family, mask in conditions(data, selected):
            pred = predictor.forward(indices, mask, details=True)
            outputs = {'original_rm': (pred['core_probs'], pred['core_y']),
                       'original_rm_lora_fusion_bias': (pred['probs'], pred['y'])}
            for name, (probability, intensity) in outputs.items():
                if not np.isfinite(probability).all() or not np.isfinite(intensity).all():
                    raise RuntimeError('Nonfinite frozen prediction')
                np.testing.assert_allclose(probability.sum(1), 1, atol=1e-6)
                result = metrics(data['y'], probability, intensity)
                rows.append(dict(split=split, scenario=scenario, family=family, model=name,
                                 n=len(indices), sample_limited=args.limit is not None,
                                 **{k: result[k] for k in ('accuracy','macro_f1','mae','corr','corr_defined')}))
                np.savez_compressed(output/'predictions'/f'{split}_{scenario}_{name}.npz',
                                    ids=data['ids'], probabilities=probability, intensity=intensity)
            if weights is not None and family in ('clean', 'mixed'):
                final_cm = grouped_confusion(labels, pred['probs'].argmax(1), group_index, n_groups)
                core_cm = grouped_confusion(labels, pred['core_probs'].argmax(1), group_index, n_groups)
                headline_cms[scenario] = (final_cm, core_cm)
                final_class, core_class = pred['probs'].argmax(1), pred['core_probs'].argmax(1)
                n01 = int(((core_class != labels) & (final_class == labels)).sum())
                n10 = int(((core_class == labels) & (final_class != labels)).sum())
                mcnemar = float(binomtest(n01,n01+n10,.5).pvalue) if n01+n10 else 1.
                statistics.append(dict(split=split, scenario=scenario, model='original_rm_lora_fusion_bias',
                                       n_videos=n_groups, metrics=paired_comparison(final_cm,core_cm,weights,signs),
                                       mcnemar_exact_p=mcnemar, baseline_wrong_new_right=n01,
                                       baseline_right_new_wrong=n10))
            print(f'{split}/{scenario}: {len(indices)} samples, frozen RM and final fusion', flush=True)
        if 'clean' in headline_cms and 'mixed40' in headline_cms:
            final_cms = [headline_cms[s][0] for s in ['clean','mixed40']]
            core_cms = [headline_cms[s][1] for s in ['clean','mixed40']]
            statistics.append(dict(split=split, scenario='mean_clean_mixed40',
                model='original_rm_lora_fusion_bias', n_videos=n_groups,
                metrics=paired_mean_views(final_cms,core_cms,weights,signs),
                aggregation='Arithmetic mean of two view metrics inside each paired video resample'))
    primary = [row for row in statistics if row['split']=='test' and row['scenario'] in ['clean','mixed40']]
    if len(primary) == 2:
        adjusted = holm([row['metrics']['macro_f1']['cluster_permutation_p_two_sided'] for row in primary])
        for row,value in zip(primary,adjusted):
            row['test_two_headlines_holm_p'] = float(value)
    for filename, chosen in [('all_metrics.csv',rows), ('headline_metrics.csv',[r for r in rows if r['family'] != 'rate'])]:
        if not chosen:
            continue
        with (output/filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(chosen[0]))
            writer.writeheader(); writer.writerows(chosen)
    write_json(output/'statistics.json', statistics)
    write_json(output/'protocol.json', protocol)
    write_json(output/'complete.json', dict(passed=True, metric_rows=len(rows),
               sample_limited=args.limit is not None, selection_performed=False,
               protocol_sha256=digest(output/'protocol.json')))
    print(f'Reevaluation outputs: {output}', flush=True)


if __name__ == '__main__':
    main()
