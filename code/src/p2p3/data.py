"""Aligned input handling. Raw zero detection ALWAYS precedes normalization."""
import json
import os
import pickle
from pathlib import Path
import numpy as np

from project_paths import ROOT, PROJECT_ROOT, DATA_ROOT, PAPER_ROOT, resolve_asset, relative_asset
OUT = ROOT / 'outputs/problem2'
CACHE_ROOT = Path(os.environ.get('COLING_CACHE_ROOT', OUT / 'cache')).resolve()
MODALITIES = ('text', 'audio', 'vision')


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # NumPy 2 pickles can be loaded by older NumPy installations.
        if int(np.__version__.split('.')[0]) < 2:
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_pickle(path):
    with open(path, 'rb') as f:
        return CompatibleUnpickler(f).load()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def zero_runs(z):
    z = np.asarray(z, dtype=bool)
    d = np.diff(np.r_[False, z, False].astype(int))
    return [[int(s), int(e)] for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


def unwrap(d):
    return d['test'] if 'test' in d else d


def lengths_and_padding(d):
    if 'text_bert' in d:
        t = np.asarray(d['text_bert'])
        if t.ndim == 2:
            t = t[None]
        # Last marked position, rather than sum: preserve holes in a damaged mask.
        alive = (t[:, 1] != 0) | (t[:, 0] != 0)
        length = (alive * np.arange(1, 51)).max(1)
    else:
        alive = None
        for m in MODALITIES:
            x = np.asarray(d[m]); x = x[None] if x.ndim == 2 else x
            nz = np.any(np.nan_to_num(x, nan=0, posinf=0, neginf=0) != 0, axis=-1)
            alive = nz if alive is None else alive | nz
        length = (alive * np.arange(1, 51)).max(1)
    # A completely empty input has unknown length. Retain 50 locations, report ambiguity.
    length = np.where(length > 0, length, 50).astype(np.int64)
    return length, np.arange(50)[None] < length[:, None]


def raw_arrays(d, text_features=None):
    d = unwrap(d)
    lengths, padding = lengths_and_padding(d)
    arrays, obs = [], []
    for m in MODALITIES:
        x = text_features if m == 'text' and text_features is not None else d[m]
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2: x = x[None]
        x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
        o = np.any(x != 0, -1) & padding
        x = x * o[..., None]
        arrays.append(x); obs.append(o)
    return arrays, np.stack(obs, 1), lengths


def load_split(split, mmap=False):
    path = CACHE_ROOT / f'{split}.npz'
    if not path.is_file():
        raise FileNotFoundError(f'Missing prepared split {path}; run src/p2p3/prepare.py from code/.')
    d = np.load(path, allow_pickle=False, mmap_mode='r' if mmap else None)
    return {k: d[k] for k in d.files}


def normalize(arrays, obs, stats):
    return [np.clip((x - np.asarray(stats[m]['mean'], np.float32)) /
                    np.asarray(stats[m]['std'], np.float32), -5, 5) * obs[:, j, :, None]
            for j, (m, x) in enumerate(zip(MODALITIES, arrays))]
