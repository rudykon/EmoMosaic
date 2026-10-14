"""Verify frozen workflow interfaces; write new results outside frozen records."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

from data import ROOT
from project_paths import DATA_ROOT, PAPER_ROOT
from improvement_common import digest, save_json
from improvement_evaluation import CachedEvaluator
from predict_workflow import WorkflowPredictor
from train import validation_masks
from workflow_oof import RUN
from workflow_experiment import collect, ensure_freeze, rules
from workflow_models import predict_candidate


def output_directory(value=None):
    """Keep verification/report exports separate from research evidence and paper."""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(value or ROOT/'outputs/reproduced'/f'workflow_{stamp}').resolve()
    for protected in (ROOT/'outputs/problem1', ROOT/'outputs/problem2',
                      ROOT/'outputs/problem3', ROOT/'models', DATA_ROOT, PAPER_ROOT,
                      ROOT/'src'):
        if output == protected.resolve() or protected.resolve() in output.parents:
            raise ValueError(f'Output would overwrite preserved project assets: {output}')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    output = output_directory(args.output_root)
    connect = socket.socket.connect; connect_ex = socket.socket.connect_ex
    def guarded(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6): raise RuntimeError('Offline verification forbids network')
        return connect(self, address)
    def guarded_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6): raise RuntimeError('Offline verification forbids network')
        return connect_ex(self, address)
    socket.socket.connect = guarded; socket.socket.connect_ex = guarded_ex
    torch.set_num_threads(2)
    frozen = ensure_freeze(); valid = json.loads((RUN/'validation_metrics.json').read_text())
    names = {'baseline', frozen['selected']}
    for kind in ['bias_shrink', 'adaptive', 'review']:
        names.add(max([r for r in valid if r['kind'] == kind], key=lambda r: r['mean_macro_f1'])['name'])
    candidates = [c for c in json.loads((RUN/'candidates.json').read_text()) if c['name'] in names]
    source = ROOT/'src/p2p3'
    save_json(output/'runtime_freeze.json', {'decision_freeze_sha256': digest(RUN/'frozen.json'),
              'source_sha256': {name: digest(source/name) for name in ['predict_workflow.py', 'verify_workflow.py']}})
    evaluator = CachedEvaluator(2); data = evaluator.dataset('attachment3')
    paths = sorted((DATA_ROOT/'source_data').glob('attachment3*/aligned/*.pkl'))
    assert [p.name for p in paths] == data['files'].tolist()
    predictor = WorkflowPredictor('baseline')
    raw = [predictor.base.load_file(p) for p in paths]
    text_absent = np.ones_like(data['obs']); text_absent[:, 0] = False
    views = [('natural', None, list(range(30))),
             ('mixed40', validation_masks(data['lengths'], 20260923)[1], [0, 10, 20]),
             ('text_absent', text_absent, [0, 10, 20]),
             ('all_absent', np.zeros_like(data['obs']), [0, 10, 20])]
    rows = []; total = 0; traces = []; baseline_actual = []
    for view, mask, indices in views:
        # collect() validates saved component identities and hashes. Require both
        # files so the verification cannot generate caches inside frozen results.
        for suffix in ('.json', '.npz'):
            path = RUN/'components/attachment3'/f'{view}{suffix}'
            if not path.is_file():
                raise FileNotFoundError(f'Missing preserved workflow component: {path}')
        cached = collect(evaluator, 'attachment3', mask, view)
        for candidate in candidates:
            expected_p, expected_y, expected_trigger = predict_candidate(cached, candidate, rules())
            predictor.name = candidate['name']; predictor.candidate = candidate
            delta_p = 0.; delta_y = 0.; times = []; calls = 0
            for i in indices:
                started = time.perf_counter()
                result = predictor.predict(raw[i], None if mask is None else mask[i:i+1])
                times.append(time.perf_counter()-started)
                p = result['probabilities'][0]; y = result['intensity'][0]
                delta_p = max(delta_p, float(np.max(np.abs(p-expected_p[i]))))
                delta_y = max(delta_y, float(abs(y-expected_y[i])))
                assert p.argmax() == expected_p[i].argmax(), (view, candidate['name'], i)
                assert result['review_triggered'][0] == expected_trigger[i]
                assert len(result['actions'][0]) <= 2
                if not expected_trigger[i]: assert not result['actions'][0]
                if view in ['text_absent', 'all_absent']: assert not result['actions'][0]
                np.testing.assert_allclose(p.sum(), 1., atol=1e-7)
                np.testing.assert_allclose(result['core_contributions'].sum(1)+result['core_interaction'],
                                           result['core_intensity'], atol=1e-6)
                calls += len(result['actions'][0]); total += 1
                traces.append({'candidate': candidate['name'], 'view': view, 'file': paths[i].name,
                               'actions': result['actions'][0]})
                if view == 'natural' and candidate['name'] == 'baseline': baseline_actual.append(p)
            assert delta_p < .002 and delta_y < .002, (view, candidate['name'], delta_p, delta_y)
            row = {'view': view, 'candidate': candidate['name'], 'n': len(indices),
                   'max_probability_difference': delta_p, 'max_intensity_difference': delta_y,
                   'class_disagreements': 0, 'total_extra_tools': calls,
                   'median_seconds_per_sample': float(np.median(times)), 'total_seconds': sum(times)}
            rows.append(row); print(row, flush=True)
    save_json(output/'raw_interface_verification.json', {'passed': True, 'raw_prediction_cases': total,
              'network_disabled': True, 'network_guard': 'Python IPv4/IPv6 connect and connect_ex blocked',
              'checks': rows, 'max_extra_tools': 2, 'no_text_disables_review': True,
              'attribution_scope': 'original RM core intensity only',
              'input_sha256': {p.name: digest(p) for p in paths}})
    save_json(output/'action_traces.json', traces)
    print('Offline raw workflow verification passed:', total, 'cases', flush=True)
    print('New verification results:', output, flush=True)


if __name__ == '__main__': main()
