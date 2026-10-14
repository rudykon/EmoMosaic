"""Verify plotted layered-card values, source timing and feature provenance."""
import hashlib
import json
import numpy as np
from project_paths import resolve_asset
from data import ROOT, MODALITIES, write_json
from p3_common import OUT3, digest


def main():
    load = lambda p: json.loads(p.read_text())
    selection = load(OUT3 / 'cards/selection.json')['samples']
    with np.load(ROOT / 'outputs/problem2/cache/valid.npz') as cache:
        valid = {k: cache[k] for k in ['ids', 'audio', 'vision', 'obs']}
    checked = []
    for c in selection:
        path = OUT3 / 'samples' / c['split'] / (c['id'] + '.json')
        s = load(path); r = load(OUT3 / 'cards' / (c['figure'] + '.json'))
        assert r['source_sample_sha256'] == digest(path)
        assert r['split'] == c['split'] and r['id'] == c['id']
        assert r['local_target'] == 'original_predicted_class_probability'
        for j, track in enumerate(r['tracks']):
            original = s['positions'][j]
            expected = {p['position'] for p in original if p['eligible'] and p.get('word_index') is not None}
            seen = []
            for row in track['values']:
                if c['split'] == 'attachment4':
                    positions = row['positions']; representative = row['representative_position']
                    assert representative in positions
                    for i in positions:
                        assert original[i]['word_index'] == row['word_index']
                        assert original[i]['time_start'] == row['time_start']
                        assert original[i]['time_end'] == row['time_end']
                    best = max(abs(original[i]['class_probability_delta']) for i in positions)
                    assert abs(row['delta']) == best
                    assert row['timing_usable'] == original[representative]['timing_usable']
                    seen.extend(positions)
                else:
                    representative = row['position']; seen.append(representative)
                assert row['delta'] == original[representative]['class_probability_delta']
            assert len(seen) == len(set(seen)) and set(seen) == expected
            maximum = max((abs(v['delta']) for v in track['values']), default=0) or 1e-6
            assert track['scale'] == maximum
        if c['split'] == 'attachment4':
            assert r['axis'] == 'source_seconds'
            aligned_path = OUT3 / 'alignment' / (c['id'] + '.json')
            aligned = load(aligned_path)
            assert digest(resolve_asset(s['video'])) == r['source_sha256'] == aligned['video_sha256']
            assert digest(aligned_path) == r['alignment_sha256']
            assert digest(OUT3 / 'alignment' / (c['id'] + '.npz')) == r['signals_sha256']
            assert r['duration'] == aligned['stream']['duration']
            assert r['waveform_samples'] == aligned['stream']['decoded_audio_samples']
            assert r['timing_usable_fraction'] == aligned['timing_usable_fraction']
            for frame in r['frames']:
                assert abs(frame['decoded_pts_absolute'] - aligned['stream']['common_start'] - frame['pts']) < 1e-8
                for pos in frame['roles']:
                    assert any(e['position'] == pos for e in s['top_evidence']['vision'])
                    keyframe = s['positions'][2][pos]['keyframe']
                    assert keyframe['frame_index'] == frame['frame_index']
                    assert keyframe['pts_relative'] == frame['pts']
            assert any(not f['roles'] for f in r['frames'])
            if c['id'] in ['15', '16']:
                assert r['timing_usable_fraction'] == 0
                assert not any(a['timing_usable'] for a in r['anchors'])
        else:
            assert r['axis'] == 'official_token_position' and not r['raw_media_available']
            assert s['video'] is None
            ix = list(valid['ids']).index(c['id'])
            start, end = min(r['positions']), max(r['positions'])
            for m in ['audio', 'vision']:
                data = valid[m][ix, start:end + 1].T.copy(); item = r['features'][m]
                assert item['shape'] == list(data.shape)
                assert hashlib.sha256(data.tobytes()).hexdigest() == item['feature_sha256']
                np.testing.assert_array_equal(item['observed'], valid['obs'][ix, MODALITIES.index(m), start:end + 1])
        for suffix in ['.pdf', '.png']:
            assert (OUT3 / 'figures' / (c['figure'] + suffix)).is_file()
        checked.append(dict(split=c['split'], id=c['id'], figure=c['figure'], axis=r['axis'],
                            pdf_sha256=digest(OUT3 / 'figures' / (c['figure'] + '.pdf'))))
    result = dict(passed=True, cards=len(checked), temporal_cards=sum(c['split']=='attachment4' for c in checked),
                  token_axis_cards=sum(c['split']=='valid' for c in checked), cards_checked=checked,
                  source_frame_pts_verified=True, waveform_lengths_verified=True,
                  complete_eligible_position_coverage=True, low_trust_timing_preserved=True,
                  input_feature_provenance_verified=True, signed_effects_exact=True)
    write_json(OUT3 / 'paper_checks/layered_cards_verification.json', result)
    print('Verified', len(checked), 'layered cards: source timing, signs, positions and features.', flush=True)


if __name__ == '__main__':main()
