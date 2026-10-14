"""Shared P3 protocol. No fitting, label lookup, or modification of P2 assets."""
import hashlib
import os
import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from project_paths import DATA_ROOT
from data import ROOT, MODALITIES, load_split, load_pickle, unwrap, write_json

OUT3 = Path(os.environ.get('COLING_P3_OUTPUT_ROOT', ROOT / 'outputs/problem3')).resolve()
SEED = 20260924
LABELS = ['Negative', 'Neutral', 'Positive']


def require_legacy_protocol():
    """Historical scripts must never replace final-fusion artifacts."""
    path = OUT3 / 'explanation_rules.json'
    if path.exists() and __import__('json').loads(path.read_text()).get('schema') == 'p3_final_fusion_v1':
        raise RuntimeError('This is a historical RM-only script. Use run_p3_fusion.py for the final RM + BERT LoRA model. '
                           'The historical RM results archive has been removed; use an isolated historical workspace for old scripts.')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def attachment_paths():
    return sorted((DATA_ROOT / 'source_data').glob('attachment4*/*/aligned/*.pkl'))


def dataset(split):
    path = OUT3 / 'cache' / f'{split}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            d = {k:z[k] for k in z.files}
    else:
        d = load_split(split)
    if split == 'attachment4':
        paths = attachment_paths()
        assert [p.name for p in paths] == d['files'].tolist()
        d['raw_text'] = np.asarray([str(unwrap(load_pickle(p))['raw_text']) for p in paths])
        # The distributed float16 cache is the canonical offline inference input.
        # Match Predictor.load_file's inverse/forward normalization arithmetic.
        from predict import Predictor
        p = Predictor(device='cpu')
        rows = [p.input_tensors(p.load_file(path))[0][0].numpy()[0] for path in paths]
        d['text'] = np.stack(rows)
    return d


def tensors(d, device, indices=None):
    sl = slice(None) if indices is None else indices
    return ([torch.as_tensor(d[m][sl], device=device, dtype=torch.float32) for m in MODALITIES],
            torch.as_tensor(d['obs'][sl], device=device),
            torch.as_tensor(d['lengths'][sl], device=device))


def select_tensors(t, ids):
    return [v[ids] for v in t[0]], t[1][ids], t[2][ids]


def eligible(d):
    return d['obs'] & (~np.isin(d['token_ids'], [0, 101, 102]))[:, None, :]


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def fusion(o, g, a, obs):
    out = np.zeros_like(o, dtype=np.float32)
    for n in range(len(o)):
        for m in range(3):
            mask = obs[n, m]
            if mask.any():
                out[n, m, mask] = np.mean([
                    rankdata(np.abs(v[n, m, mask]), method='average') / mask.sum()
                    for v in (o, g, a)], axis=0)
    return out


def scores(e, obs, method):
    if method == 'fusion':
        return fusion(e['occlusion'], e['gradient'], e['alpha'], obs)
    return np.abs(e[method]) * obs


def ordering(score, keep):
    """Stable position tie break; unavailable positions never enter an order."""
    ids = np.flatnonzero(keep)
    return ids[np.argsort(-score[ids], kind='stable')]


@torch.inference_mode()
def masked_predictions(predictor, t, masks, indices=None, batch_size=192, per_seed=False):
    masks = np.asarray(masks, bool)
    if indices is None:
        indices = np.arange(len(masks))
    result = []
    for s in range(0, len(masks), batch_size):
        ix = torch.as_tensor(indices[s:s+batch_size], device=t[1].device)
        xx, oo, ll = select_tensors(t, ix)
        mm = torch.as_tensor(masks[s:s+batch_size], device=oo.device)
        ys = torch.stack([model(xx, oo, ll, mm)['y'] for model in predictor.models])
        result.append((ys if per_seed else ys.mean(0)).cpu().numpy())
    return np.concatenate(result, axis=1 if per_seed else 0)


def token_mapping(raw, ids, tokenizer):
    enc = tokenizer(raw, truncation=True, max_length=50, padding='max_length',
                    return_offsets_mapping=True)
    if not np.array_equal(ids, enc['input_ids']):
        raise ValueError('Frozen tokenizer IDs disagree with official input; cannot assign offsets')
    words = list(re.finditer(r'\S+', raw))
    out = []
    for i, (tok, (u, v)) in enumerate(zip(ids, enc['offset_mapping'])):
        wi = next((j for j, w in enumerate(words) if w.start() < v and w.end() > u), None)
        w = words[wi] if wi is not None else None
        out.append(dict(position=i, token=tokenizer.convert_ids_to_tokens(int(tok)),
                        token_id=int(tok), char_start=int(u), char_end=int(v),
                        word_index=wi, word=w.group() if w else None,
                        word_char_start=w.start() if w else None,
                        word_char_end=w.end() if w else None,
                        context=raw[max(0, u-30):min(len(raw), v+30)] if w else '',
                        special=bool(tok in (0, 101, 102))))
    return out


def provenance():
    folder = ROOT / 'models/rmnet'
    paths = [folder / 'decision_rules.json', folder / 'norm_stats.json', ROOT / 'src/p2p3/config.yaml']
    paths += sorted(folder.glob('weights_seed*_fp16.pt'))
    paths += [ROOT / 'outputs/problem2/cache/attachment4_text_cache.npz']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}
