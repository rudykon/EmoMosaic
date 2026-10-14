"""Offline raw-file inference with the validation-frozen local workflow.

The original default predictor stays intact. All actions consume only observed
inputs and frozen parameters; no label, search, or fitting occurs here.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import ROOT, MODALITIES, unwrap
from improved_predict import ImprovedPredictor
from lora_text_expert import load_adapter, predict_adapter
from workflow_oof import RUN, tool_mask
from workflow_models import quality, review_trigger, predict_candidate
from workflow_experiment import rules, ensure_freeze, canonical_tools


class WorkflowPredictor:
    def __init__(self, candidate=None, verify=True):
        if verify: ensure_freeze()
        selected = json.loads((RUN/'selected.json').read_text())['name']
        self.name = candidate or selected
        candidates = json.loads((RUN/'candidates.json').read_text())
        self.candidate = next(c for c in candidates if c['name'] == self.name)
        self.rules = rules()
        self.base = ImprovedPredictor(ROOT/'models/rmnet_bert_lora/decision_rules.json')
        self.adapters = [load_adapter(ROOT/p, ROOT/self.rules['bert_model_dir'])
                         for p in self.rules['expert']['adapters']]

    @torch.inference_mode()
    def predict(self, raw, mask=None):
        data = dict(unwrap(raw))
        xx, obs, lengths = self.base.original.input_tensors(data)
        tokens = np.asarray(data['text_bert']); tokens = tokens[None] if tokens.ndim == 2 else tokens
        prepared = dict(zip(MODALITIES, [v.cpu().numpy() for v in xx]),
                        obs=obs.numpy(), lengths=lengths.numpy(), token_ids=tokens[:, 0])
        keep = np.ones_like(prepared['obs']) if mask is None else np.asarray(mask, dtype=bool)
        if keep.ndim == 2: keep = keep[None]
        if keep.shape != prepared['obs'].shape: raise ValueError('Invalid workflow observation mask')
        parts = [m(xx, obs, lengths, torch.as_tensor(keep)) for m in self.base.models]
        rp = torch.stack([p['logits'].softmax(-1) for p in parts]).mean(0).numpy()
        ry = torch.stack([p['y'] for p in parts]).mean(0).numpy()
        lp = [predict_adapter(adapter, prepared, keep) for adapter in self.adapters]
        c = {'rm_p': rp, 'rm_y': ry, 'lora_p': np.mean([p['probabilities'] for p in lp], 0),
             'lora_y': np.mean([p['intensity'] for p in lp], 0), 'quality': quality(prepared, keep)[0]}
        actions = [[] for _ in range(len(ry))]
        trigger = review_trigger(c, self.rules) if self.candidate['kind'] == 'review' else np.zeros(len(ry), bool)
        for tool in ['av', 'text']:
            c[tool+'_p'] = rp.copy(); c[tool+'_y'] = ry.copy()
            available = c['quality'][:, 0] > 0 if tool == 'text' else np.maximum(c['quality'][:, 1], c['quality'][:, 2]) > 0
            ix = np.flatnonzero(trigger & available)
            if not len(ix): continue
            reduced = torch.as_tensor(tool_mask(keep[ix], tool))
            results = [m([x[ix] for x in xx], obs[ix], lengths[ix], reduced) for m in self.base.models]
            c[tool+'_p'][ix] = torch.stack([p['logits'].softmax(-1) for p in results]).mean(0).numpy()
            c[tool+'_y'][ix] = torch.stack([p['y'] for p in results]).mean(0).numpy()
            for i in ix: actions[i].append('rm_'+tool)
        canonical_tools(c)
        p, y, actual_trigger = predict_candidate(c, self.candidate, self.rules)
        np.testing.assert_array_equal(actual_trigger, trigger)
        assert max(map(len, actions), default=0) <= 2
        contribution = torch.stack([p['contributions'] for p in parts]).mean(0).numpy()
        interaction = torch.stack([p['y_int'] for p in parts]).mean(0).numpy()
        return {'probabilities': p, 'intensity': y, 'core_intensity': ry,
                'core_contributions': contribution, 'core_interaction': interaction,
                'review_triggered': trigger, 'actions': actions, 'components': c}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output', required=True); ap.add_argument('--candidate')
    ap.add_argument('--threads', type=int, default=2); args = ap.parse_args()
    torch.set_num_threads(args.threads)
    predictor = WorkflowPredictor(args.candidate); rows = []
    for path in sorted(Path(args.input_dir).glob('*.pkl')):
        started = time.perf_counter(); raw = predictor.base.load_file(path)
        p = predictor.predict(raw); prob = p['probabilities'][0]
        rows.append({'file': path.name, 'sample_id': str(raw.get('id', path.stem)),
                     'pred_polarity': ['Negative', 'Neutral', 'Positive'][prob.argmax()],
                     'pred_intensity': float(p['intensity'][0]), 'p_neg': float(prob[0]),
                     'p_neu': float(prob[1]), 'p_pos': float(prob[2]),
                     'core_intensity': float(p['core_intensity'][0]),
                     'core_contribution_text': float(p['core_contributions'][0, 0]),
                     'core_contribution_audio': float(p['core_contributions'][0, 1]),
                     'core_contribution_vision': float(p['core_contributions'][0, 2]),
                     'core_interaction': float(p['core_interaction'][0]),
                     'attribution_target': 'original_rm_core_intensity', 'workflow': predictor.name,
                     'review_triggered': bool(p['review_triggered'][0]),
                     'review_actions': json.dumps(p['actions'][0]),
                     'seconds': time.perf_counter()-started})
    if not rows: raise ValueError('No input files')
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {len(rows)} offline workflow predictions ({predictor.name}): {output}', flush=True)


if __name__ == '__main__': main()
