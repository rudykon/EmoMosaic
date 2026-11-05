"""Workflow contracts: missing-input fallback, bounded actions, feature isolation."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data import ROOT
from workflow_models import (baseline, diagnostic_features, quality, review_trigger,
                             predict_gate, predict_candidate, fit_gate)
from workflow_experiment import canonical_tools
from project_paths import DATA_ROOT, PAPER_ROOT
from verify_workflow import output_directory
from model_paths import asset_matches


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.rules = json.loads((ROOT/'models/rmnet_bert_lora/decision_rules.json').read_text())
        rng = np.random.default_rng(43); n = 15
        self.c = {'quality': rng.uniform(.3, 1, (n, 10)),
                  **{k+'_p': rng.dirichlet([2, 2, 2], n) for k in ['rm', 'lora', 'av', 'text']},
                  **{k+'_y': rng.uniform(-2, 2, n) for k in ['rm', 'lora', 'av', 'text']}}

    def gate(self, review=False):
        width = diagnostic_features(self.c, review).shape[1]
        shape = (width+1, 4) if review else (width+1,)
        return {'kind': 'review_gate' if review else 'adaptive_gate', 'mean': [0.]*width,
                'scale': [1.]*width, 'class_weights': np.zeros(shape).tolist(),
                'regression_weights': np.zeros(shape).tolist(), 'bias': self.rules['fusion']['bias']}

    def test_zero_adaptive_gate_recovers_frozen_baseline(self):
        a, b = baseline(self.c, self.rules)
        p, y = predict_gate(self.c, self.gate(), self.rules)
        np.testing.assert_allclose(p, a, atol=1e-12)
        np.testing.assert_allclose(y, b, atol=1e-12)

    def test_text_absence_disables_expert_and_review(self):
        self.c['quality'][:, 0] = 0
        expected = baseline(self.c, self.rules)
        for review in [False, True]:
            p, y = predict_gate(self.c, self.gate(review), self.rules)
            np.testing.assert_allclose(p, expected[0]); np.testing.assert_allclose(y, self.c['rm_y'])
        self.assertFalse(review_trigger(self.c, self.rules).any())

    def test_nontriggered_rows_are_exact_adaptive_output(self):
        self.c['rm_p'][:] = [.97, .02, .01]; self.c['lora_p'][:] = [.98, .01, .01]
        gate = self.gate()
        candidate = {'kind': 'review', 'adaptive': gate, 'gate': self.gate(True)}
        p, y, trigger = predict_candidate(self.c, candidate, self.rules)
        expected = predict_gate(self.c, gate, self.rules)
        self.assertFalse(trigger.any()); np.testing.assert_array_equal(p, expected[0])
        np.testing.assert_array_equal(y, expected[1])

    def test_unavailable_tool_has_canonical_features(self):
        self.c['quality'][:, 1:3] = 0
        canonical_tools(self.c)
        np.testing.assert_array_equal(self.c['av_p'], self.c['rm_p'])
        np.testing.assert_array_equal(self.c['av_y'], self.c['rm_y'])

    def test_quality_uses_masks_and_unknown_without_targets(self):
        d = {'obs': np.ones((1, 3, 50), bool), 'lengths': np.array([5]),
             'token_ids': np.array([[101, 123, 100, 456, 102]+[0]*45]), 'y': np.array([2.])}
        d['obs'][:, :, 5:] = False; mask = np.ones_like(d['obs']); mask[:, 0, 3] = False
        a, names = quality(d, mask); self.assertAlmostEqual(a[0, 0], 1/3)
        d['y'][:] = -3; b, _ = quality(d, mask)
        np.testing.assert_array_equal(a, b); self.assertEqual(len(names), a.shape[1])

    def test_fitted_gates_are_finite_simplex_and_bounded(self):
        y = np.linspace(-2, 2, len(self.c['rm_y']))
        for review in [False, True]:
            gate = fit_gate(self.c, y, .1, self.rules, review)
            p, r = predict_gate(self.c, gate, self.rules)
            self.assertTrue(np.isfinite(p).all() and np.isfinite(r).all())
            np.testing.assert_allclose(p.sum(1), 1)
            self.assertTrue((p >= 0).all() and (np.abs(r) <= 3).all())
            self.assertEqual(gate['fit_rows'], len(y))

    def test_exports_cannot_target_preserved_assets(self):
        for folder in [ROOT/'outputs/problem2/workflow_experiment', ROOT/'models',
                       ROOT/'src', DATA_ROOT, PAPER_ROOT]:
            with self.subTest(folder=folder):
                with self.assertRaises(ValueError):
                    output_directory(folder/'new_workflow_report')

    def test_source_migration_requires_exact_verified_chain(self):
        sha = lambda value: hashlib.sha256(value.encode()).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current = root/'src/p2p3/worker.py'
            current.parent.mkdir(parents=True)
            current.write_text('portable source')
            snapshot = root/'outputs/provenance/parent.py'
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text('parent source')
            prior = root/'models/rmnet_bert_lora/path_migration.json'
            prior.parent.mkdir(parents=True)
            prior.write_text(json.dumps({'files': [{
                'path': 'src/P2P3_model/worker.py',
                'before_sha256': sha('historical source'),
                'after_sha256': sha('parent source')}]}))
            (snapshot.parent/'standalone_source_migration.json').write_text(json.dumps({'files': [{
                'path': 'src/p2p3/worker.py',
                'before_snapshot': 'outputs/provenance/parent.py',
                'before_sha256': sha('parent source'),
                'after_sha256': sha('portable source')}]}))
            historical = root/'src/P2P3_model/worker.py'
            self.assertTrue(asset_matches(historical, sha('historical source'), root))
            self.assertFalse(asset_matches(historical, sha('unrelated source'), root))
            snapshot.write_text('changed parent source')
            self.assertFalse(asset_matches(historical, sha('historical source'), root))
            snapshot.write_text('parent source')
            current.write_text('changed portable source')
            self.assertFalse(asset_matches(historical, sha('historical source'), root))


if __name__ == '__main__': unittest.main()
