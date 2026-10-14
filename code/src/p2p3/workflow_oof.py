"""Video-disjoint RM predictions paired with the existing train-only LoRA OOF.

No validation/test labels are read by the fold workers. Fold preprocessing is
fitted from unnormalized features; normalized/clipped full-train caches cannot
be inverted safely and are deliberately not used to fit fold statistics.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from project_paths import DATA_ROOT
from data import ROOT, OUT, MODALITIES, load_pickle, load_split, raw_arrays, normalize
from improvement_common import EXP, digest, save_json
from model_paths import asset_matches, current_model_paths
from missing import training_mask
from model import RMNet
from train import device_data, infer, validation_masks

RUN = OUT / 'workflow_experiment'


def initialize():
    RUN.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT/'models/rmnet_bert_lora/decision_rules.json').read_text())
    lora = EXP/'lora/group_cv/train_group_oof.npz'
    config = yaml.safe_load((ROOT/'src/p2p3/config.yaml').read_text())
    epochs = int(np.median([json.loads((OUT/f'runs/A0_s{s}/complete.json').read_text())['best_epoch']
                            for s in [17, 29, 43, 71, 101]]))
    protocol = {
        'version': 1, 'date': '2026-09-27', 'baseline': old,
        'baseline_config_sha256': digest(ROOT/'models/rmnet_bert_lora/decision_rules.json'),
        'fit_split': 'attachment2/train', 'selection_split': 'attachment2/valid',
        'test_policy': 'Historically exposed; freeze all candidates before descriptive test evaluation; no test selection',
        'primary_metric': 'mean(valid_clean_macro_f1, valid_mixed40_macro_f1)',
        'folds': 5, 'fold_assignment': 'existing video-group LoRA OOF; all RM fits exclude the matching video groups',
        'rm_seed': 17, 'rm_epochs': epochs, 'rm_epoch_rule': 'median historical validation-selected A0 epochs',
        'rm_config': config, 'rm_scheduler_total_epochs': config['epochs'],
        'normalization': 'refit on fold training rows from unnormalized frozen BERT last states and raw audio/vision',
        'lora_oof_sha256': digest(lora),
        'lora_oof_protocol_sha256': digest(EXP/'lora/group_cv/protocol.json'),
        'oof_scope': 'single-seed RM and single-seed LoRA OOF; deployment uses frozen five-seed ensembles',
        'oof_limitation': 'Conditional on existing validation-chosen base architectures/durations, not nested unbiased CV',
        'views': ['clean', 'mixed40'], 'mask_seed': 20260923,
        'missing_semantics': 'existing encode-then-feature-mask protocol, not strict raw-token deletion',
        'candidates': {'baseline': True, 'bias_shrink_factors': [0.0, 0.5],
                       'adaptive_l2': [0.01, 0.1, 1.0], 'review_l2': [0.01, 0.1, 1.0]},
        'review_trigger': 'text present AND (RM/LoRA argmax disagree OR baseline top-two margin < 0.20)',
        'review_tools': ['RM audio+vision only, if either observed', 'RM text only, if text observed'],
        'max_extra_tools': 2,
        'controller': 'bounded diagnostic workflow with learned fusion; no generative language-model agent',
        'promotion': {'min_valid_mean_f1_gain': 0.003, 'min_each_view_f1_gain': 0.0,
                      'max_each_view_mae_increase': 0.01, 'tie_break': 'fewer extra tools, then lexical name'},
        'uncertainty': 'paired original-video resampling; 2000 replicates, seed 20260927',
        'source_sha256': {'rm_model': digest(ROOT/'src/p2p3/model.py'),
                          'fold_worker': digest(Path(__file__)),
                          'lora_oof_worker': digest(ROOT/'src/p2p3/cross_validate_lora.py')},
    }
    path = RUN/'protocol.json'
    if path.exists():
        previous = json.loads(path.read_text())
        comparable = current_model_paths(previous)
        if asset_matches(ROOT/'models/rmnet_bert_lora/decision_rules.json', previous['baseline_config_sha256']):
            comparable['baseline_config_sha256'] = protocol['baseline_config_sha256']
        if asset_matches(Path(__file__), previous['source_sha256']['fold_worker']):
            comparable['source_sha256']['fold_worker'] = protocol['source_sha256']['fold_worker']
        if comparable != protocol:
            raise RuntimeError('Existing workflow protocol differs; preserve it')
        return
    save_json(path, protocol)
    print('Workflow protocol frozen; RM epochs:', epochs, flush=True)


def prepare_raw():
    folder = RUN/'raw_train'; marker = folder/'complete.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        for name, expected in record['files_sha256'].items():
            assert digest(folder/name) == expected
        return
    folder.mkdir(parents=True, exist_ok=True)
    source = DATA_ROOT/'source_data/attachment2_features/aligned_50.pkl'
    supplied = load_pickle(source)['train']
    cached = load_split('train')
    np.testing.assert_array_equal(np.asarray(supplied['id'], dtype=str), cached['ids'])
    text_source = EXP/'features/bert/train/hidden.npy'
    text_manifest = json.loads((text_source.parent/'complete.json').read_text())
    assert digest(text_source) == text_manifest['hidden_sha256']
    frozen = np.load(text_source, mmap_mode='r')
    text = np.asarray(frozen[:, -1], np.float32)
    arrays, obs, lengths = raw_arrays(supplied, text)
    np.testing.assert_array_equal(obs, cached['obs'])
    np.testing.assert_array_equal(lengths, cached['lengths'])
    for name, values in zip(MODALITIES, arrays):
        np.save(folder/f'{name}.npy', values)
    np.savez(folder/'metadata.npz', obs=obs, lengths=lengths, y=cached['y'],
             ids=cached['ids'], token_ids=cached['token_ids'])
    save_json(marker, {'source_sha256': digest(source), 'frozen_text_sha256': digest(text_source),
                      'files_sha256': {p.name: digest(p) for p in folder.iterdir() if p.is_file()}})
    print('Unnormalized training features verified and prepared', flush=True)


def tool_mask(mask, tool):
    result = mask.copy()
    if tool == 'av': result[:, 0] = False
    elif tool == 'text': result[:, 1:] = False
    elif tool != 'full': raise ValueError(tool)
    return result


def fit_fold(fold, threads):
    torch.set_num_threads(threads)
    protocol = json.loads((RUN/'protocol.json').read_text())
    folder = RUN/'folds'/f'fold{fold}'; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists():
        record = json.loads((folder/'complete.json').read_text())
        assert record['protocol_sha256'] == digest(RUN/'protocol.json')
        assert record['prediction_sha256'] == digest(folder/'predictions.npz')
        print('Already complete fold', fold, flush=True); return
    cfg = protocol['rm_config']; seed = protocol['rm_seed']; epochs = protocol['rm_epochs']
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with np.load(EXP/'lora/group_cv/train_group_oof.npz') as z:
        ids = z['ids']; groups = z['groups']; fold_id = z['fold_id']
    with np.load(RUN/'raw_train/metadata.npz') as z: meta = {k: z[k] for k in z.files}
    np.testing.assert_array_equal(ids, meta['ids'])
    held = np.flatnonzero(fold_id == fold); fit = np.flatnonzero(fold_id != fold)
    assert not set(groups[fit]) & set(groups[held])
    arrays = [np.load(RUN/f'raw_train/{m}.npy', mmap_mode='r') for m in MODALITIES]
    stats = {}
    for j, (name, values) in enumerate(zip(MODALITIES, arrays)):
        rows = values[fit][meta['obs'][fit, j]].astype(np.float64)
        stats[name] = {'mean': rows.mean(0).tolist(), 'std': np.maximum(rows.std(0), 1e-5).tolist(),
                       'n_observed': len(rows)}
    save_json(folder/'norm_stats.json', stats)
    normalized = normalize(arrays, meta['obs'], stats)
    d = dict(zip(MODALITIES, normalized), **meta)
    tr = {k: v[fit] for k, v in d.items()}; va = {k: v[held] for k, v in d.items()}
    xt, ot, lt = device_data(tr, 'cpu'); tv = device_data(va, 'cpu')
    y = torch.as_tensor(tr['y']); cy = (y.sign()+1).long()
    cw = 1/torch.bincount(cy, minlength=3).float(); cw /= cw.mean()
    model = RMNet(cfg, 'A0')
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    steps = math.ceil(len(y)/cfg['batch_size']); total = cfg['epochs']*steps
    warm = max(1, int(total*cfg['warmup_fraction']))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: (step+1)/warm if step < warm else
                                             .5*(1+math.cos(math.pi*(step-warm)/max(1, total-warm))))
    started = time.time(); history = []
    for epoch in range(epochs):
        model.train(); order = rng.permutation(len(y)); losses = []
        for start in range(0, len(y), cfg['batch_size']):
            ix = order[start:start+cfg['batch_size']]
            keep = torch.as_tensor(training_mask(tr['lengths'][ix], rng, epoch, cfg['epochs'], cfg=cfg))
            x = [v[ix] for v in xt]; o = ot[ix]; length = lt[ix]; target = y[ix]
            opt.zero_grad(set_to_none=True)
            clean = model(x, o, length); p = model(x, o, length, keep)
            task = F.l1_loss(clean['y'], target)+F.l1_loss(p['y'], target)
            present = p['obs'].any(-1)
            uni = ((p['y_m']-target[:, None]).abs()*present).sum()/present.sum().clamp_min(1)
            ce = (F.cross_entropy(p['logits'], cy[ix], weight=cw)+F.cross_entropy(clean['logits'], cy[ix], weight=cw))/2
            rec = torch.zeros(())
            for j in range(3):
                hidden = o[:, j] & ~p['obs'][:, j]
                err = F.smooth_l1_loss(p['recon'][j], x[j], reduction='none').mean(-1)
                rec += (err*hidden).sum()/hidden.sum().clamp_min(1)/3
            con = F.mse_loss(p['z'], clean['z'].detach())+F.l1_loss(p['y'], clean['y'].detach())
            loss = task+cfg['lambda_unimodal']*uni+cfg['lambda_class']*ce+cfg['lambda_reconstruction']*rec
            loss = loss+cfg['lambda_consistency']*con+cfg['lambda_interaction']*p['y_int'].abs().mean()
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite fold training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
            opt.step(); sched.step(); losses.append(float(loss.detach()))
        row = {'epoch': epoch+1, 'loss': float(np.mean(losses)), 'seconds': time.time()-started}
        history.append(row); save_json(folder/'history.json', history)
        print(f'Fold {fold} epoch {epoch+1}/{epochs}: loss={row["loss"]:.4f}, {row["seconds"]:.1f}s', flush=True)
    # The fixed last epoch is used; no held-out labels, validation, or test read.
    torch.save({'config': cfg, 'variant': 'A0', 'state_dict': model.state_dict(),
                'seed': seed, 'epochs': epochs}, folder/'model.pt')
    masks = [np.ones_like(meta['obs']), validation_masks(meta['lengths'], protocol['mask_seed'])[1]]
    prediction = {'ids': ids[held], 'indices': held, 'groups': groups[held]}
    for tool in ['full', 'av', 'text']:
        outputs = [infer(model, tv, torch.as_tensor(tool_mask(m[held], tool))) for m in masks]
        prediction[tool+'_p'] = np.array([torch.from_numpy(p['logits']).softmax(-1).numpy() for p in outputs])
        prediction[tool+'_y'] = np.array([p['y'] for p in outputs])
    np.savez_compressed(folder/'predictions.npz', **prediction)
    save_json(folder/'complete.json', {'fold': fold, 'fit_n': len(fit), 'held_n': len(held),
              'video_overlap': 0, 'epochs': epochs, 'seconds': time.time()-started,
              'fit_ids_sha256': __import__('hashlib').sha256(ids[fit].tobytes()).hexdigest(),
              'held_ids_sha256': __import__('hashlib').sha256(ids[held].tobytes()).hexdigest(),
              'normalization_sha256': digest(folder/'norm_stats.json'),
              'model_sha256': digest(folder/'model.pt'), 'prediction_sha256': digest(folder/'predictions.npz'),
              'protocol_sha256': digest(RUN/'protocol.json')})
    print('Completed fold', fold, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--init', action='store_true')
    ap.add_argument('--prepare', action='store_true'); ap.add_argument('--fold', type=int)
    ap.add_argument('--threads', type=int, default=2); args = ap.parse_args()
    if args.init: initialize()
    if args.prepare: prepare_raw()
    if args.fold is not None: fit_fold(args.fold, args.threads)
