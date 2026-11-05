"""Shared, validation-only selection and masked fusion for the P2 exploration."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.special import softmax

from data import OUT, MODALITIES
from metrics import classification
from train import validation_masks

EXP = OUT / 'improvement'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Independent experiment queues may refresh the same readiness report.
    # Each writer owns its temporary file; readers always see a complete JSON.
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    tmp=Path(name)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def masked(data, mask=None):
    obs = data['obs'] if mask is None else data['obs'] & mask
    return dict(data, obs=obs, **{m: np.where(obs[:, j, :, None], data[m], 0)
                                for j, m in enumerate(MODALITIES)})


def views(data):
    return [masked(data), masked(data, validation_masks(data['lengths'], 20260923)[1])]


def pool_tokens(x, observed, mode='mean_max_min'):
    """No missing/padding token participates, including endpoints and extrema."""
    x = np.asarray(x, dtype=np.float32)
    keep = np.asarray(observed, dtype=bool)[..., None]
    count = keep.sum(1).clip(1)
    mean = np.where(keep, x, 0).sum(1) / count
    if mode == 'mean':
        return mean.astype(np.float32)
    high = np.where(keep, x, -np.inf).max(1)
    low = np.where(keep, x, np.inf).min(1)
    empty = ~keep.any(1)
    high = np.where(empty, 0, high)
    low = np.where(empty, 0, low)
    return np.concatenate([mean, high, low], axis=-1).astype(np.float32)


def metrics(y, probabilities, intensity=None):
    result = classification(y, np.asarray(probabilities).argmax(-1))
    if intensity is not None:
        result['mae'] = float(np.mean(np.abs(y - intensity)))
        defined = bool(np.std(intensity) > 1e-7 and np.std(y) > 1e-7)
        result['corr_defined'] = defined
        result['corr'] = float(np.corrcoef(y, intensity)[0, 1]) if defined else 0.
    return result


def score_views(y, probabilities):
    return float(np.mean([metrics(y, p)['macro_f1'] for p in probabilities]))


def apply_bias(probabilities, bias):
    return softmax(np.log(np.clip(probabilities, 1e-12, 1)) + np.asarray(bias), axis=-1)


def select_bias(y, probabilities):
    """Mean of per-view F1, not F1 of concatenated views; deterministic ties."""
    labels = np.sign(y).astype(int) + 1
    scores = np.log(np.clip(np.asarray(probabilities), 1e-12, 1))
    best_score, best_bias = score_views(y, probabilities), np.zeros(3)
    for step, radius in [(0.1, 1.), (.025, .1)]:
        center = best_bias.copy()
        grid = [np.array([center[0] + a, 0, center[2] + b])
                for a in np.arange(-radius, radius + step / 2, step)
                for b in np.arange(-radius, radius + step / 2, step)]
        # Evaluate a modest batch of bias candidates at once.
        for start in range(0, len(grid), 64):
            biases = np.asarray(grid[start:start+64])
            pred = (scores[None] + biases[:, None, None]).argmax(-1)
            f1 = np.zeros(pred.shape[:2])
            for cls in range(3):
                tp = ((pred == cls) & (labels[None, None] == cls)).sum(-1)
                den = (pred == cls).sum(-1) + (labels == cls).sum()
                f1 += 2 * tp / np.maximum(den, 1) / 3
            values = f1.mean(-1)
            for value, bias in zip(values, biases):
                if value > best_score + 1e-12:
                    best_score, best_bias = float(value), bias.copy()
    return {'score': best_score, 'bias': best_bias.tolist()}


def fusion(expert, core, observed_text_fraction, w0, exponent, bias=(0., 0., 0.)):
    fraction = np.clip(np.asarray(observed_text_fraction), 0, 1)
    weight = w0 * fraction ** exponent
    # Explicit guard also handles exponent=0: absent text must disable the expert.
    weight = np.where(fraction > 0, weight, 0)
    p = weight[:, None] * expert + (1 - weight[:, None]) * core
    return apply_bias(p, bias), weight


def text_observed_fraction(data):
    """BERT content slots only: supplied [UNK] carries no lexical information."""
    ids=np.asarray(data['token_ids'])
    content=(ids!=0)&(ids!=101)&(ids!=102)
    informative=content&(ids!=100)&data['obs'][:,0]
    return informative.sum(1)/content.sum(1).clip(1)


def initialize_protocol():
    path = EXP / 'protocol.json'
    protocol = {
        'version': 1,
        'primary_metric': 'mean(valid_clean_macro_f1, valid_mixed40_macro_f1)',
        'labels': 'negative: y<0, neutral: y==0, positive: y>0',
        'fit_splits': ['train'], 'selection_splits': ['valid'],
        'test_policy': 'Prior test results are known; new candidates are frozen before one final evaluation. No test-based selection.',
        'mask_seed': 20260923, 'comparison_grid': {'types': ['T','A','V','TA','TV','AV','TAV'], 'rates': [.2,.4,.6], 'replay_patterns': 30},
        'probe_C': [.001,.003,.01,.03], 'ridge_alpha': [10.,100.,1000.],
        'classification_weights': [.3,1.,2.],
        'training_seeds': [17,29,43,71,101,113,127,139,151,163,179,191,211,227,241],
        'bert_layers': list(range(13)), 'bert_pools': ['mean','mean_max_min'],
        'qwen_model': 'Qwen/Qwen2.5-1.5B', 'qwen_revision': '8faed761d45a263340a0528343f099c05c9a4323',
        'qwen_layers': list(range(4,29,2)), 'qwen_pool': 'mean_max_min',
        'qwen_larger_model': 'Qwen/Qwen2.5-7B',
        'expert_gates': {'enter_fusion_clean_valid_f1': .62, 'strong_signal': .66},
        'fusion_weights': np.linspace(0,1,11).tolist(), 'fusion_exponents': [1.,2.,4.],
        'cv': '5 folds grouped by original video inside train; valid used only for fixed configuration selection',
        'uncertainty': '95% paired bootstrap by video, exact McNemar accuracy test, disclose historical test exposure',
        'stage4': 'Original statement does not require frozen backbones; consider train-only fine-tuning after frozen mainline, log decision and evidence.',
        'deliverables': ['validation_search','rmnet_45_runs','bert_13_layer_scalar_mix','llm_probes','reliability_fusion',
                         'strict_delete_reencode','train_group_cv','frozen_test_grid','attachment3_csv','predict_interface',
                         'model_revision_cache_sha256','paper_ablation_intervals_mcnemar','p3_scope_disclosure','reproduction_package'],
    }
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError('Protocol exists and differs; use a versioned explicit amendment')
    else:
        save_json(path, protocol)
    return protocol
