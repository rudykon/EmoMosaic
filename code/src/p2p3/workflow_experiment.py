"""Fit, freeze, and evaluate the RM+LoRA diagnostic workflows separately."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

from data import ROOT, OUT, load_split
from improvement_common import EXP, digest, save_json, metrics
from improvement_evaluation import CachedEvaluator
from train import device_data, infer, validation_masks
from model import RMNet
from workflow_oof import RUN, tool_mask
from model_paths import asset_matches, current_model_paths
from workflow_models import (quality, baseline, review_trigger, slice_components,
                             fit_gate, predict_candidate)


def rules():
    return current_model_paths(json.loads((RUN/'protocol.json').read_text())['baseline'])


def experiment_config():
    config = rules(); config['base_dir'] = str(ROOT)
    return config


def canonical_tools(c):
    # An unavailable tool has no evidence. Its canonical feature values are
    # the already available core output, and its fusion weight is zero.
    for name, present in [('text', c['quality'][:, 0] > 0),
                          ('av', np.maximum(c['quality'][:, 1], c['quality'][:, 2]) > 0)]:
        c[name+'_p'][~present] = c['rm_p'][~present]
        c[name+'_y'][~present] = c['rm_y'][~present]
    return c


def ensure_freeze():
    f = json.loads((RUN/'frozen.json').read_text())
    for name, expected in f['protected_sha256'].items():
        if not asset_matches(ROOT/name, expected): raise RuntimeError('Frozen workflow artifact changed: '+name)
    return f


def collect(evaluator, split, mask, name):
    if split not in ['train', 'valid']: ensure_freeze()
    folder = RUN/'components'/split; folder.mkdir(parents=True, exist_ok=True)
    path = folder/f'{name}.npz'; manifest = folder/f'{name}.json'
    d = evaluator.dataset(split)
    keep = np.ones_like(d['obs']) if mask is None else np.asarray(mask, dtype=bool)
    identity = {'input_sha256': evaluator.file_digest(OUT/f'cache/{split}.npz'),
                'protocol_sha256': digest(RUN/'protocol.json'),
                'mask_sha256': __import__('hashlib').sha256(keep.tobytes()).hexdigest()}
    if manifest.exists():
        saved = json.loads(manifest.read_text()); assert saved['identity'] == identity
        assert digest(path) == saved['sha256']
        with np.load(path) as z:
            np.testing.assert_array_equal(z['ids'], d['ids'])
            return {k: z[k] for k in z.files if k != 'ids'}
    config = experiment_config()
    started = time.time(); base = evaluator.core(config, split, keep)
    lp, ly = evaluator.expert(config, split, keep)
    c = {'quality': quality(d, keep)[0], 'rm_p': base['probabilities'], 'rm_y': base['intensity'],
         'lora_p': lp, 'lora_y': ly}
    trigger = review_trigger(c, rules()); calls = {}
    for tool in ['av', 'text']:
        c[tool+'_p'] = c['rm_p'].copy(); c[tool+'_y'] = c['rm_y'].copy()
        available = (c['quality'][:, 0] > 0 if tool == 'text' else
                     np.maximum(c['quality'][:, 1], c['quality'][:, 2]) > 0)
        ix = np.flatnonzero(trigger & available); calls[tool] = len(ix)
        if not len(ix): continue
        subset = {k: v[ix] for k, v in d.items()}
        tensors = device_data(subset, 'cpu'); mask_tool = torch.as_tensor(tool_mask(keep[ix], tool))
        pp = []; yy = []
        for checkpoint in config['core_checkpoints']:
            full = ROOT/checkpoint; key = (str(full), evaluator.file_digest(full))
            if key not in evaluator.models:
                saved = torch.load(full, map_location='cpu', weights_only=False)
                model = RMNet(saved['config'], saved['variant'])
                model.load_state_dict(saved['state_dict']); model.eval()
                evaluator.models[key] = model
            model = evaluator.models[key]
            output = infer(model, tensors, mask_tool)
            pp.append(softmax(output['logits'], axis=-1)); yy.append(output['y'])
        c[tool+'_p'][ix] = np.mean(pp, axis=0); c[tool+'_y'][ix] = np.mean(yy, axis=0)
    canonical_tools(c)
    np.savez_compressed(path, ids=d['ids'], **c)
    save_json(manifest, {'identity': identity, 'sha256': digest(path), 'n': len(d['ids']),
              'triggered': int(trigger.sum()), 'tool_samples': calls, 'seconds': time.time()-started,
              'execution': 'tools executed only on triggered samples with corresponding observed inputs'})
    print(f'Collected {split}/{name}: review={trigger.sum()}/{len(trigger)}, tools={calls}', flush=True)
    return c


def collect_valid(threads):
    evaluator = CachedEvaluator(threads); d = evaluator.dataset('valid')
    return [collect(evaluator, 'valid', m, name) for name, m in
            [('clean', None), ('mixed40', validation_masks(d['lengths'], 20260923)[1])]]


def training_components():
    protocol = json.loads((RUN/'protocol.json').read_text())
    path = EXP/'lora/group_cv/train_group_oof.npz'
    assert digest(path) == protocol['lora_oof_sha256']
    d = load_split('train')
    with np.load(path) as z:
        np.testing.assert_array_equal(z['ids'], d['ids'])
        lp, ly, fold_ids, groups = z['probabilities'], z['intensity'], z['fold_id'], z['groups']
    models = {name: np.zeros((2, len(d['ids']), 3)) for name in ['full_p', 'av_p', 'text_p']}
    models.update({name: np.zeros((2, len(d['ids']))) for name in ['full_y', 'av_y', 'text_y']})
    seen = np.zeros(len(d['ids']), dtype=int)
    for fold in range(5):
        folder = RUN/'folds'/f'fold{fold}'
        record = json.loads((folder/'complete.json').read_text())
        assert record['video_overlap'] == 0 and record['protocol_sha256'] == digest(RUN/'protocol.json')
        assert record['prediction_sha256'] == digest(folder/'predictions.npz')
        with np.load(folder/'predictions.npz') as z:
            ix = z['indices']; assert np.all(fold_ids[ix] == fold)
            np.testing.assert_array_equal(z['ids'], d['ids'][ix])
            assert not set(groups[ix]) & set(groups[fold_ids != fold])
            for key in models: models[key][:, ix] = z[key]
            seen[ix] += 1
    np.testing.assert_array_equal(seen, 1)
    masks = [np.ones_like(d['obs']), validation_masks(d['lengths'], 20260923)[1]]
    pieces = []
    for view, mask in enumerate(masks):
        c = {'quality': quality(d, mask)[0], 'lora_p': lp[view], 'lora_y': ly[view],
             'rm_p': models['full_p'][view], 'rm_y': models['full_y'][view],
             'av_p': models['av_p'][view], 'av_y': models['av_y'][view],
             'text_p': models['text_p'][view], 'text_y': models['text_y'][view]}
        pieces.append(canonical_tools(c))
    merged = {k: np.concatenate([c[k] for c in pieces]) for k in pieces[0]}
    np.savez_compressed(RUN/'train_oof_components.npz', ids=np.tile(d['ids'], 2),
                        fold_id=np.tile(fold_ids, 2), **merged)
    return merged, np.tile(d['y'], 2)


def summarize(candidate, components, target):
    scored = []; predictions = []
    for c in components:
        p, y, trigger = predict_candidate(c, candidate, rules())
        row = metrics(target, p, y)
        row['review_fraction'] = float(trigger.mean())
        row['mean_extra_tools'] = float(np.mean(trigger.astype(int)*(1+(
            np.maximum(c['quality'][:, 1], c['quality'][:, 2]) > 0))))
        scored.append(row); predictions.append((p, y))
    return {'name': candidate['name'], 'kind': candidate['kind'], 'clean': scored[0], 'mixed40': scored[1],
            'mean_macro_f1': float(np.mean([r['macro_f1'] for r in scored])),
            'mean_mae': float(np.mean([r['mae'] for r in scored]))}, predictions


def fit(threads):
    if (RUN/'frozen.json').exists():
        ensure_freeze(); print('Already frozen; refusing reselection', flush=True); return
    torch.set_num_threads(threads)
    protocol = json.loads((RUN/'protocol.json').read_text())
    design = {
        'classification_objective': 'cross entropy weighted by inverse square root training OOF class frequency',
        'regression_objective': '0.3 * Huber(beta=0.5)',
        'regularization': 'lambda * mean squared gate coefficients + 0.1 * mean squared bias parameters',
        'bias': 'frozen baseline bias + 0.5*tanh(train-fitted delta)',
        'optimizer': 'float64 LBFGS, max_iter 180, strong_wolfe; zero coefficient initialization',
        'feature_scaling': 'OOF training mean/std, std lower bound 0.05',
        'missing_fallback': 'text completely absent => frozen core probability bias and unmodified core intensity',
        'adaptive_prior': {'classification_text_weight': 0.8, 'regression_text_weight': 0.6},
        'review_prior_order': ['RM full', 'LoRA', 'RM AV', 'RM text'],
        'review_prior': {'classification': [.15, .75, .05, .05], 'regression': [.30, .60, .05, .05]},
        'features_source_sha256': digest(ROOT/'src/p2p3/workflow_models.py'),
    }
    if (RUN/'controller_design.json').exists():
        assert json.loads((RUN/'controller_design.json').read_text()) == design
    save_json(RUN/'controller_design.json', design)
    c, y = training_components()
    candidates = [{'name': 'baseline', 'kind': 'baseline'}]
    candidates += [{'name': f'bias_shrink_{v:g}', 'kind': 'bias_shrink', 'shrink': v}
                   for v in protocol['candidates']['bias_shrink_factors']]
    for strength in protocol['candidates']['adaptive_l2']:
        gate = fit_gate(c, y, strength, rules())
        candidates.append({'name': f'adaptive_l2_{strength:g}', 'kind': 'adaptive', 'gate': gate})
        print('Fitted adaptive', strength, gate['objective'], flush=True)
    trigger = review_trigger(c, rules())
    for strength in protocol['candidates']['review_l2']:
        gate = fit_gate(slice_components(c, trigger), y[trigger], strength, rules(), review=True)
        adaptive = next(v for v in candidates if v['name'] == f'adaptive_l2_{strength:g}')['gate']
        candidates.append({'name': f'review_l2_{strength:g}', 'kind': 'review', 'gate': gate, 'adaptive': adaptive})
        print('Fitted review', strength, gate['objective'], flush=True)
    save_json(RUN/'candidates.json', candidates)
    components = collect_valid(threads); valid = load_split('valid')
    rows = []; folder = RUN/'valid_predictions'; folder.mkdir(exist_ok=True)
    for candidate in candidates:
        row, pred = summarize(candidate, components, valid['y']); rows.append(row)
        np.savez_compressed(folder/(candidate['name']+'.npz'), ids=valid['ids'],
                            probabilities=np.array([p for p, _ in pred]), intensity=np.array([y for _, y in pred]))
        print('Validation', row['name'], f'{row["mean_macro_f1"]:.6f}', flush=True)
    baseline_row = rows[0]; gate = protocol['promotion']
    eligible = [r for r in rows[1:] if r['mean_macro_f1']-baseline_row['mean_macro_f1'] >= gate['min_valid_mean_f1_gain']
                and all(r[v]['macro_f1'] >= baseline_row[v]['macro_f1']+gate['min_each_view_f1_gain']
                        and r[v]['mae'] <= baseline_row[v]['mae']+gate['max_each_view_mae_increase']
                        for v in ['clean', 'mixed40'])]
    ranked = sorted(eligible, key=lambda r: (-r['mean_macro_f1'], r['clean']['mean_extra_tools']+r['mixed40']['mean_extra_tools'], r['name']))
    selected = ranked[0]['name'] if ranked else 'baseline'
    save_json(RUN/'validation_metrics.json', rows)
    save_json(RUN/'selected.json', {'name': selected, 'promoted_by_validation': selected != 'baseline',
              'eligible': [r['name'] for r in ranked], 'best_unrestricted': max(rows, key=lambda r: r['mean_macro_f1'])['name'],
              'decision_uses': ['valid'], 'test_read_for_selection': False,
              'candidate': next(c for c in candidates if c['name'] == selected)})
    protected = [RUN/'protocol.json', RUN/'controller_design.json', RUN/'candidates.json', RUN/'validation_metrics.json', RUN/'selected.json',
                 RUN/'train_oof_components.npz', ROOT/'models/rmnet_bert_lora/decision_rules.json']
    protected += [ROOT/p for p in rules()['core_checkpoints']+rules()['expert']['adapters']]
    protected += [ROOT/'models/rmnet/norm_stats.json', ROOT/'models/pretrained/bert/model.safetensors']
    protected += [ROOT/'src/p2p3'/n for n in ['workflow_oof.py', 'workflow_models.py', 'workflow_experiment.py',
                  'model.py', 'lora_text_expert.py', 'improvement_common.py', 'train.py']]
    save_json(RUN/'frozen.json', {'selected': selected, 'test_read_for_selection': False,
              'historical_test_exposure': True, 'protected_sha256': {str(p.relative_to(ROOT)): digest(p) for p in protected}})
    print('Frozen selection:', selected, flush=True)


def evaluate(threads):
    frozen = ensure_freeze()
    candidates = json.loads((RUN/'candidates.json').read_text())
    evaluator = CachedEvaluator(threads); d = evaluator.dataset('test')
    components = [collect(evaluator, 'test', m, name) for name, m in
                  [('clean', None), ('mixed40', validation_masks(d['lengths'], 20260923)[1])]]
    rows = []; predictions = {}; folder = RUN/'test_predictions'; folder.mkdir(exist_ok=True)
    for candidate in candidates:
        row, pred = summarize(candidate, components, d['y']); rows.append(row)
        predictions[candidate['name']] = np.array([p.argmax(-1) for p, _ in pred])
        np.savez_compressed(folder/(candidate['name']+'.npz'), ids=d['ids'],
                            probabilities=np.array([p for p, _ in pred]), intensity=np.array([y for _, y in pred]))
        print('Frozen test', row['name'], f'{row["mean_macro_f1"]:.6f}', flush=True)
    save_json(RUN/'test_metrics.json', rows)
    from analyze_boosted_comparison import grouped_confusion
    from evaluate_improvement import paired_mean_views
    groups = np.unique([str(i).split('$_$')[0] for i in d['ids']], return_inverse=True)[1]
    n = int(groups.max()+1); rng = np.random.default_rng(20260927)
    weights = rng.multinomial(n, np.repeat(1/n, n), size=2000)
    signs = rng.choice([-1, 1], size=(2000, n)); truth = np.sign(d['y']).astype(int)+1
    base = [grouped_confusion(truth, p, groups, n) for p in predictions['baseline']]
    stats = {}
    for name, pred in predictions.items():
        cm = [grouped_confusion(truth, p, groups, n) for p in pred]
        stats[name] = paired_mean_views(cm, base, weights, signs)
    save_json(RUN/'test_statistics.json', {'unit': 'original video', 'n_videos': n, 'replicates': 2000,
              'interpretation': 'descriptive, historically exposed test; candidate-wise unadjusted intervals', 'comparisons': stats})
    save_json(RUN/'evaluation_complete.json', {'frozen_sha256': digest(RUN/'frozen.json'),
              'selected': frozen['selected'], 'test_changed_selection': False, 'candidates': len(candidates),
              'test_samples': len(d['ids']), 'test_sha256': digest(RUN/'test_metrics.json')})


def robustness(threads):
    frozen = ensure_freeze()
    evaluator = CachedEvaluator(threads); d = evaluator.dataset('test')
    candidates = json.loads((RUN/'candidates.json').read_text())
    # Compare baseline and the validation-best member of each prespecified family.
    valid = json.loads((RUN/'validation_metrics.json').read_text())
    names = {'baseline', frozen['selected']}
    for kind in ['bias_shrink', 'adaptive', 'review']:
        names.add(max([r for r in valid if r['kind'] == kind], key=lambda r: r['mean_macro_f1'])['name'])
    candidates = [c for c in candidates if c['name'] in names]
    from missing import fixed_mask, TYPES
    masks = [(f'{kind}_{rate:g}', fixed_mask(d['lengths'], kind, rate, 20260923))
             for kind in TYPES for rate in [.2, .4, .6]]
    empty = np.zeros_like(d['obs']); absent_text = np.ones_like(d['obs']); absent_text[:, 0] = False
    masks += [('text_absent', absent_text), ('all_absent', empty)]
    rows = []
    for name, mask in masks:
        c = collect(evaluator, 'test', mask, name)
        for candidate in candidates:
            p, y, review = predict_candidate(c, candidate, rules())
            row = dict(scenario=name, model=candidate['name'], **metrics(d['y'], p, y))
            row['review_fraction'] = float(review.mean()); rows.append(row)
    save_json(RUN/'robustness_metrics.json', rows)
    print('Robustness scenarios completed:', len(masks), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--phase', choices=['valid', 'fit', 'evaluate', 'robustness'], required=True)
    ap.add_argument('--threads', type=int, default=2); args = ap.parse_args()
    {'valid': collect_valid, 'fit': fit, 'evaluate': evaluate, 'robustness': robustness}[args.phase](args.threads)
