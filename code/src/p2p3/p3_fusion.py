"""Frozen RM + BERT LoRA inference for P3 output-level interventions.

Matches ImprovedPredictor: encode text once, mask contextual states at pooling,
recompute text reliability and BOTH fusion paths after every intervention.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data import ROOT, MODALITIES, write_json
from improved_predict import ImprovedPredictor
from lora_text_expert import load_adapter
from p3_common import digest, dataset, eligible, OUT3

CONFIG = ROOT / 'models/rmnet_bert_lora/decision_rules.json'
SEED = 20260927
LABELS = np.asarray(['Negative', 'Neutral', 'Positive'])


def provenance():
    cfg = json.loads(CONFIG.read_text())
    files = [CONFIG, ROOT/'models/rmnet/norm_stats.json',
             ROOT/'models/pretrained/bert/model.safetensors',
             *[ROOT/p for p in cfg['core_checkpoints']],
             *[ROOT/p for p in cfg['expert']['adapters']]]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


class FusionExplainer:
    def __init__(self, device='cpu', batch_size=128):
        self.public = ImprovedPredictor(CONFIG, device)
        self.cfg = self.public.config
        if self.cfg['expert']['kind'] != 'bert_lora' or self.public.layer_mix:
            raise ValueError('P3 requires the frozen original RM + BERT LoRA configuration')
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.adapters = []
        for name in self.cfg['expert']['adapters']:
            m = load_adapter(self.public.resolve(name), self.public.resolve(self.cfg['bert_model_dir']))
            self.adapters.append(m.to(self.device).eval().requires_grad_(False))
        for m in self.public.models:
            m.requires_grad_(False)

    def prepare(self, data, name):
        """FP32 hidden cache is an exact factorization of deployed pooling masks."""
        self.data = data
        self.x = [torch.as_tensor(data[m], dtype=torch.float32, device=self.device) for m in MODALITIES]
        self.obs = torch.as_tensor(data['obs'], device=self.device, dtype=torch.bool)
        self.ids = torch.as_tensor(data['token_ids'], device=self.device, dtype=torch.long)
        self.lengths = torch.as_tensor(data['lengths'], device=self.device)
        self.hidden = []
        cache = OUT3 / 'fusion_cache'; cache.mkdir(exist_ok=True)
        for j, model in enumerate(self.adapters):
            h = hashlib.sha256()
            h.update(np.asarray(data['token_ids']).tobytes())
            h.update(np.asarray(data['obs'][:, 0]).tobytes())
            h.update(digest(self.public.resolve(self.cfg['expert']['adapters'][j])).encode())
            # Frozen base fingerprint recorded in the global manifest.
            h.update(b'fp32-contextual-v1')
            path = cache / f'{name}_{j}_{h.hexdigest()[:20]}.npy'
            if not path.exists():
                states = []
                with torch.inference_mode():
                    for s in range(0, len(self.ids), 24):
                        states.append(model.encode(self.ids[s:s+24], self.obs[s:s+24,0]).cpu().numpy())
                np.save(path, np.concatenate(states))
                print('encoded', name, 'adapter', j, flush=True)
            self.hidden.append(np.load(path, mmap_mode='r'))
        return self

    @torch.inference_mode()
    def forward(self, indices, masks=None, details=False):
        indices = np.asarray(indices, dtype=np.int64)
        masks = None if masks is None else np.asarray(masks, dtype=bool)
        if masks is not None and masks.shape != (len(indices), 3, 50):
            raise ValueError('Masks must have shape [interventions, 3, 50]')
        results = []
        for s in range(0, len(indices), self.batch_size):
            ix = indices[s:s+self.batch_size]
            oo = self.obs[ix]
            mm = torch.ones_like(oo) if masks is None else torch.as_tensor(masks[s:s+len(ix)], device=self.device)
            effective = oo & mm
            parts = [m([v[ix] for v in self.x], oo, self.lengths[ix], mm) for m in self.public.models]
            cy = torch.stack([p['y'] for p in parts]).mean(0)
            cp = torch.stack([p['logits'].softmax(-1) for p in parts]).mean(0)
            ids = self.ids[ix]
            content = (ids != 0) & (ids != 101) & (ids != 102)
            f = ((content & (ids != 100) & effective[:,0]).sum(1).double()
                 / content.sum(1).clamp_min(1))
            keep = effective[:,0] & (ids != 100)
            expert = [m.classify(torch.as_tensor(np.array(h[ix]), device=self.device), keep)
                      for m, h in zip(self.adapters, self.hidden)]
            ep = torch.stack([p['logits'].softmax(-1) for p in expert]).mean(0)
            ey = torch.stack([p['y'] for p in expert]).mean(0)
            cf = self.cfg['fusion']
            wc = torch.where(f > 0, cf['w0'] * f**cf['k'], 0.)
            wr = torch.where(f > 0, self.cfg['regression_w0'] * f**self.cfg['regression_k'], 0.)
            mixed = wc[:,None]*ep + (1-wc[:,None])*cp
            bias = torch.tensor(cf['bias'], device=self.device, dtype=torch.float64)
            probs = (mixed.clamp(1e-12, 1).log()+bias).softmax(-1)
            out = {'y': wr*ey+(1-wr)*cy, 'probs': probs}
            if details:
                contributions = torch.stack([p['contributions'] for p in parts]).mean(0)
                interaction = torch.stack([p['y_int'] for p in parts]).mean(0)
                out.update(core_y=cy, expert_y=ey, core_probs=cp, expert_probs=ep,
                           classification_weight=wc, regression_weight=wr,
                           core_gate=torch.stack([p['w'] for p in parts]).mean(0),
                           scaled_core_contributions=(1-wr[:,None])*contributions,
                           scaled_core_interaction=(1-wr)*interaction,
                           expert_contribution=wr*ey)
            results.append({k:v.cpu().numpy() for k,v in out.items()})
        return {k:np.concatenate([r[k] for r in results]) for k in results[0]}


def shapley(values):
    """[sample, subset, optional class] -> [sample, modality, optional class]."""
    result = np.zeros((len(values), 3, *values.shape[2:]), dtype=np.float64)
    for m in range(3):
        for s in range(8):
            if not s & (1 << m):
                result[:,m] += (1/3 if s.bit_count() in (0,2) else 1/6) * (values[:,s|(1<<m)]-values[:,s])
    return result


def explain(p, split):
    folder = OUT3/'explanations'; folder.mkdir(exist_ok=True)
    path = folder/f'{split}_fusion.npz'
    if path.exists():
        with np.load(path) as z:
            return dict(z)
    n = len(p.ids); base = p.forward(np.arange(n), details=True)
    masks = np.zeros((8, n, 3, 50), dtype=bool)
    for s in range(8):
        for m in range(3):
            masks[s,:,m] = bool(s & (1<<m))
    values = p.forward(np.tile(np.arange(n),8), masks.reshape(-1,3,50))
    base['subset_y'] = values['y'].reshape(8,n).T
    base['subset_probs'] = values['probs'].reshape(8,n,3).transpose(1,0,2)
    base['shapley_y'] = shapley(base['subset_y'])
    base['shapley_probs'] = shapley(base['subset_probs'])
    oy = np.zeros((n,4,50)); op = np.zeros((n,4,50,3))
    # Batch by original samples; absent slots have exactly zero effect.
    for start in range(0,n,16):
        indices=[]; mm=[]; locations=[]
        for i in range(start,min(start+16,n)):
            for scope in range(4):
                observed = p.data['obs'][i,scope] if scope < 3 else p.data['obs'][i].any(0)
                for pos in np.flatnonzero(observed):
                    mask = np.ones((3,50),bool)
                    if scope < 3: mask[scope,pos]=False
                    else: mask[:,pos]=False
                    indices.append(i); mm.append(mask); locations.append((i,scope,pos))
        out = p.forward(indices,mm)
        for j, (i,scope,pos) in enumerate(locations):
            oy[i,scope,pos] = base['y'][i]-out['y'][j]
            op[i,scope,pos] = base['probs'][i]-out['probs'][j]
        print('explain',split,min(start+16,n),'/',n,flush=True)
    base.update(occlusion_y=oy, occlusion_probs=op, ids=p.data['ids'], eligible=eligible(p.data))
    np.savez_compressed(path,**base)
    return base
