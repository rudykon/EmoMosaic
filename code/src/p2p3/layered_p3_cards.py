"""Layered evidence cards on source time or official aligned token position.

Only the presentation changes. Prediction, Shapley values and occlusion effects
are read from frozen JSON; source signals are explicitly separate from scores.
"""
import json
import subprocess
import sys
import textwrap
from functools import lru_cache

import av
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.ticker import MaxNLocator
import numpy as np

from project_paths import resolve_asset
from data import ROOT, MODALITIES, write_json
from p3_common import OUT3, digest

sys.path.insert(0, str(ROOT / 'src/common'))
from plot_style import apply_style, save_figure

FIG = OUT3 / 'figures'
POS, NEG = '#D55E00', '#0072B2'
FEATURES = LinearSegmentedColormap.from_list('layered_features', ['#762A83', '#FFFFFF', '#1B7837'])
MOD_NAMES = {'text': '文本', 'audio': '语音', 'vision': '视觉'}
LEFT, RIGHT = .13, .965


def blend(color, fraction):
    return tuple(1 - fraction * (1 - x) for x in to_rgb(color))


def group_words(sample, modality):
    """One display cell per original word; signed max-|effect| subtoken.

    This is a display representative, never an additive word attribution.
    """
    groups = {}
    for row in sample['positions'][modality]:
        if not row['eligible'] or row.get('word_index') is None:
            continue
        groups.setdefault(row['word_index'], []).append(row)
    out = []
    for rows in groups.values():
        r = max(rows, key=lambda x: (abs(x['class_probability_delta']), -x['position']))
        out.append(dict(word=r['word'], word_index=r['word_index'],
                        char_start=r['word_char_start'], char_end=r['word_char_end'],
                        positions=[x['position'] for x in rows], representative_position=r['position'],
                        delta=r['class_probability_delta'], time_start=r.get('time_start'),
                        time_end=r.get('time_end'), timing_usable=bool(r.get('timing_usable'))))
    return sorted(out, key=lambda r: r['word_index'])


def track(fig, bottom, height, xlim, label, ticks, xlabel=False):
    ax = fig.add_axes([LEFT, bottom, RIGHT - LEFT, height])
    ax.set_xlim(*xlim); ax.set_xticks(ticks)
    ax.grid(axis='x', color='#DBE0E4', lw=.6, linestyle=':')
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=8, length=2)
    ax.tick_params(axis='x', labelbottom=xlabel)
    ax.text(-.092, .5, label, transform=ax.transAxes, ha='center', va='center', fontsize=10)
    return ax


def effect_track(ax, rows, time_axis, scope):
    scale = max((abs(r['delta']) for r in rows), default=0) or 1e-6
    for r in rows:
        if time_axis:
            start, end = r['time_start'], r['time_end']
            if start is None or end is None or end <= start:
                continue
        else:
            start, end = r['position'] - .43, r['position'] + .43
        value = r['delta']; color = POS if value >= 0 else NEG
        low = time_axis and not r['timing_usable']
        ax.bar((start + end) / 2, value, width=end - start, color=blend(color, .85),
               edgecolor='#666666' if low else color, hatch='////' if low else None,
               linewidth=.35, align='center', zorder=3)
    ax.axhline(0, color='#555555', lw=.6)
    ax.set_ylim(-scale * 1.18, scale * 1.18)
    ax.set_yticks([-scale, 0, scale]); ax.set_yticklabels(['−1', '0', '+1'])
    ax.text(1.004, .5, f'×{scale:.2g}', transform=ax.transAxes, va='center', fontsize=8)
    return dict(scope=scope, scale=scale, values=rows,
                normalization='Axis ticks multiplied by the displayed per-track maximum absolute effect.')


def top_text_anchors(sample):
    result = []; seen = set()
    for r in sample['top_evidence']['text']:
        key = r['word_index']
        if key in seen or r.get('time_start') is None:
            continue
        seen.add(key)
        result.append(dict(word=r['word'], token=r['token'], position=r['position'],
                           marker=f'E{len(result)+1}',
                           start=r['time_start'], end=r['time_end'],
                           timing_usable=r['timing_usable'], delta=r['class_probability_delta']))
    return sorted(result, key=lambda r: (r['start'], r['position']))


def source_frames(sample, signal, duration):
    wanted = {}
    for r in sample['top_evidence']['vision']:
        f = r.get('keyframe')
        if f:
            item = wanted.setdefault(f['frame_index'], dict(frame_index=f['frame_index'],
                       pts=f['pts_relative'], roles=[], words=[], timing_usable=r['timing_usable']))
            item['roles'].append(r['position']); item['words'].append(r['word'])
    # Add context frames at the start and end; these are not attribution evidence.
    for t in [.08 * duration, .92 * duration]:
        k = int(np.argmin(abs(signal['source_video_pts_relative'] - t)))
        idx = int(signal['source_video_index'][k])
        wanted.setdefault(idx, dict(frame_index=idx, pts=float(signal['source_video_pts_relative'][k]),
                                   roles=[], words=[], timing_usable=None))
    images = {}
    with av.open(str(resolve_asset(sample['video']))) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                images[i] = frame.to_image()
                actual = float(frame.pts * frame.time_base)
                wanted[i]['decoded_pts_absolute'] = actual
            if i >= max(wanted):
                break
    assert set(images) == set(wanted)
    return sorted(wanted.values(), key=lambda r: (r['pts'], r['frame_index'])), images


def waveform(sample, alignment):
    meta = alignment['stream']
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(resolve_asset(sample['video'])),
               '-vn', '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1']
    y = np.frombuffer(subprocess.check_output(command), dtype='<f4')
    start = round((meta['common_start'] - meta['audio_start']) * 16000)
    y = y[start:start + int(np.floor(meta['duration'] * 16000))]
    # Preserve the min/max envelope of the actual samples; no synthesized tail.
    blocks = [y[i:i + 160] for i in range(0, len(y), 160)]
    times = np.array([(i * 160 + len(b) / 2) / 16000 for i, b in enumerate(blocks)])
    return times, np.array([b.min() for b in blocks]), np.array([b.max() for b in blocks]), len(y)


def attachment_card(sample):
    sid = sample['prediction']['id']
    aligned_path = OUT3 / 'alignment' / f'{sid}.json'
    signal_path = OUT3 / 'alignment' / f'{sid}.npz'
    alignment = json.loads(aligned_path.read_text()); signal = np.load(signal_path)
    duration = alignment['stream']['duration']; xlim = (0, duration)
    ticks = MaxNLocator(nbins=7).tick_values(*xlim); ticks = ticks[(ticks >= 0) & (ticks <= duration)]
    fig = plt.figure(figsize=(8.4, 7.56))
    anchors = top_text_anchors(sample)
    # Common evidence bands span all signal and contribution tracks.
    for a in anchors:
        x = LEFT + (RIGHT - LEFT) * a['start'] / duration
        width = (RIGHT - LEFT) * (a['end'] - a['start']) / duration
        fig.add_artist(Rectangle((x, .235), width, .464, transform=fig.transFigure,
                                 facecolor='#ECEFF1', edgecolor='#9CA6AD', lw=.5,
                                 linestyle='-' if a['timing_usable'] else '--', zorder=-1))
    text_rows = group_words(sample, 0)
    ax = track(fig, .633, .063, xlim, '文本\n候选词段', ticks)
    ax.set_ylim(0, 3); ax.set_yticks([])
    scale = max((abs(r['delta']) for r in text_rows), default=0) or 1
    for r in text_rows:
        if r['time_start'] is None or r['time_end'] <= r['time_start']:
            continue
        ax.add_patch(Rectangle((r['time_start'], .03), r['time_end'] - r['time_start'], .68,
                     facecolor=blend(POS if r['delta'] >= 0 else NEG, .12 + .75 * abs(r['delta']) / scale),
                     edgecolor='#777777', linewidth=.4, hatch=None if r['timing_usable'] else '////'))
    for i, a in enumerate(anchors):
        midpoint = (a['start'] + a['end']) / 2
        labelx = (.18 + i * .32) * duration
        ax.annotate(f"{a['marker']}  {a['word']} [{a['position']}]", xy=(midpoint, .72),
                    xytext=(labelx, 2.1), ha='center', va='center', fontsize=10,
                    arrowprops=dict(arrowstyle='-', color='#555555', lw=.7))
    ax = track(fig, .576, .041, xlim, '文本\n贡献 Δp', ticks)
    tracks = [effect_track(ax, text_rows, True, 'text')]
    time, low, high, n_samples = waveform(sample, alignment)
    ax = track(fig, .487, .068, xlim, '语音\n原始波形', ticks)
    ax.fill_between(time, low, high, color='#697681', linewidth=0)
    limit = float(max(abs(low).max(), abs(high).max())) or 1
    ax.set_ylim(-1.1 * limit, 1.1 * limit); ax.set_yticks([-limit, 0, limit])
    ax.set_yticklabels([f'{-limit:.2f}', '0', f'{limit:.2f}'])
    ax = track(fig, .418, .051, xlim, '语音\n基频 Hz', ticks)
    f0 = np.where(signal['f0'] > 0, signal['f0'], np.nan)
    ax.plot(signal['audio_times'], f0, color='#7E2F8E', lw=1.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2)); ax.set_ylim(bottom=0)
    ax = track(fig, .361, .040, xlim, '语音\n贡献 Δp', ticks)
    tracks.append(effect_track(ax, group_words(sample, 1), True, 'audio'))
    ax = track(fig, .313, .024, xlim, '视觉\n人脸检出', ticks)
    vt = signal['vision_times']; mask = signal['face_mask'].astype(float)
    ax.fill_between(vt, 0, mask, step='mid', facecolor=blend('#009E73', .4), linewidth=0)
    ax.set_ylim(0, 1); ax.set_yticks([0, 1])
    ax = track(fig, .249, .041, xlim, '视觉\n贡献 Δp', ticks, xlabel=True)
    tracks.append(effect_track(ax, group_words(sample, 2), True, 'vision'))
    ax.text(1.005, -.65, '时间 / 秒', transform=ax.transAxes, va='top', ha='right', fontsize=9)
    frame_rows, images = source_frames(sample, signal, duration)
    n = len(frame_rows); width = (RIGHT - LEFT) / n
    for i, frame in enumerate(frame_rows):
        left = LEFT + i * width + .004
        image_ax = fig.add_axes([left, .085, width - .012, .105])
        image_ax.imshow(images[frame['frame_index']]); image_ax.axis('off')
        tag = '证据' if frame['roles'] else '回看'
        positions = '/'.join(map(str, frame['roles']))
        title = f"F{i+1}  {frame['pts']:.3f}s  |  {tag}" + (f' [{positions}]' if positions else '')
        image_ax.set_title(title, fontsize=8.5, pad=4)
        image_ax.text(.5, -.10, f"原帧 {frame['frame_index']}", transform=image_ax.transAxes,
                      ha='center', va='top', fontsize=8.5)
        x = LEFT + (RIGHT - LEFT) * frame['pts'] / duration
        fig.add_artist(ConnectionPatch(xyA=(x, .237), coordsA=fig.transFigure,
                       xyB=(left + (width - .012) / 2, .203), coordsB=fig.transFigure,
                       color='#009E73' if frame['roles'] else '#929BA1', lw=.8,
                       linestyle='--' if frame['timing_usable'] is False else '-'))
    fig.text(.045, .142, '视频\n原始帧', ha='center', va='center', fontsize=10)
    usable = alignment['timing_usable_fraction']
    fig.text(.035, .715, f"共同时间轴  0–{duration:.2f}s   |   可用候选词时间 {usable:.0%}   |   WER {sample['prediction']['WER']:.3f}",
             fontsize=10, va='top')
    notes(fig, time_axis=True)
    return fig, dict(axis='source_seconds', duration=duration, anchors=anchors, tracks=tracks,
                     frames=frame_rows, waveform_samples=n_samples,
                     source_video=sample['video'], source_sha256=digest(resolve_asset(sample['video'])),
                     alignment_sha256=digest(aligned_path), signals_sha256=digest(signal_path),
                     timing_usable_fraction=usable)


@lru_cache(maxsize=1)
def validation_features():
    with np.load(ROOT / 'outputs/problem2/cache/valid.npz') as cache:
        return {key: cache[key] for key in ['ids', 'audio', 'vision', 'obs']}


def position_rows(sample, modality):
    return [dict(position=r['position'], token=r['token'], delta=r['class_probability_delta'])
            for r in sample['positions'][modality] if r['eligible']]


def validation_card(sample):
    fig = plt.figure(figsize=(8.4, 7.56))
    positions = [r['position'] for r in sample['positions'][0] if not r['special']]
    start, end = min(positions), max(positions)
    xlim = (start - .65, end + .65); ticks = positions
    ax = track(fig, .610, .094, xlim, '文本\n词元顺序', ticks)
    ax.set_ylim(0, 1); ax.set_yticks([])
    values = position_rows(sample, 0); scale = max(abs(r['delta']) for r in values) or 1
    for r in values:
        ax.add_patch(Rectangle((r['position'] - .44, 0), .88, .21,
                     facecolor=blend(POS if r['delta'] >= 0 else NEG, .15 + .7 * abs(r['delta']) / scale),
                     edgecolor='#999999', lw=.4))
        ax.text(r['position'], .23, r['token'], ha='left', va='bottom', rotation=48, fontsize=8)
    ax = track(fig, .555, .048, xlim, '文本\n贡献 Δp', ticks)
    tracks = [effect_track(ax, values, False, 'text')]
    cache = validation_features(); idx = list(cache['ids']).index(sample['prediction']['id'])
    features = {}
    for m, bottom, height, effect_bottom in [('audio', .402, .115, .331), ('vision', .204, .100, .123)]:
        j = MODALITIES.index(m); x = cache[m][idx, start:end + 1].T.copy()
        observed = cache['obs'][idx, j, start:end + 1]
        data = np.ma.array(x, mask=np.broadcast_to(~observed, x.shape))
        ax = track(fig, bottom, height, xlim, MOD_NAMES[m]+'\n输入特征', ticks)
        cmap = FEATURES.copy(); cmap.set_bad('#D8DDE0')
        ax.imshow(data, aspect='auto', interpolation='nearest', cmap=cmap, vmin=-5, vmax=5,
                  extent=(start - .5, end + .5, x.shape[0] - .5, -.5))
        ax.set_yticks([0, x.shape[0] - 1]); ax.set_yticklabels(['0', str(x.shape[0] - 1)])
        ax.text(1.003, .5, f'{x.shape[0]}维\n标准化\n[−5,5]', transform=ax.transAxes, fontsize=8, va='center')
        ax = track(fig, effect_bottom, .048, xlim, MOD_NAMES[m]+'\n贡献 Δp', ticks, xlabel=m=='vision')
        tracks.append(effect_track(ax, position_rows(sample, j), False, m))
        features[m] = dict(shape=list(x.shape), observed=observed.tolist(),
                           feature_sha256=__import__('hashlib').sha256(x.tobytes()).hexdigest())
    ax.set_xticks(positions if len(positions) < 25 else positions[::2])
    ax.set_xlabel('官方对齐序列的词元位置（不是秒数）', fontsize=10)
    notes(fig, time_axis=False)
    fig.text(.035, -.002, '共享词元轴：验证集未提供原始音视频；语音与视觉层展示官方标准化特征。灰色为未观测。',
             fontsize=8.2, va='top')
    return fig, dict(axis='official_token_position', positions=positions, tracks=tracks, features=features,
                     raw_media_available=False, feature_source='outputs/problem2/cache/valid.npz',
                     shared_axis_note_position='below_plot')


def notes(fig, time_axis):
    for x, color, text in [(.035, POS, '支持原预测类别'), (.235, NEG, '抵消原预测类别')]:
        fig.add_artist(Rectangle((x, .047), .016, .009, transform=fig.transFigure,
                                 facecolor=color, edgecolor='none'))
        fig.text(x + .023, .052, text, va='center', fontsize=9)
    fig.text(.435, .052, 'Δp = 遮挡前 − 遮挡后；各贡献轨迹旁标实际倍率。', va='center', fontsize=9)
    if time_axis:
        text = '斜纹/虚线：低可信候选时间。词级色条取绝对效应最大子词，不相加；波形、基频、人脸检出及原帧仅辅助回看。'
    else:
        text = '特征热图：紫/白/绿 = 标准化负值/零/正值；橙蓝贡献轨迹：模型遮挡效应。原有位置及观测掩码均保留。'
    fig.text(.035, .025, '\n'.join(textwrap.wrap(text, width=66)), va='center', fontsize=8.2, linespacing=1.25)


def render_card(split, sample):
    fig, record = attachment_card(sample) if split == 'attachment4' else validation_card(sample)
    name = f"card_{split}_{sample['prediction']['id']}".replace('$', '')
    save_figure(fig, FIG, name); plt.close(fig)
    record.update(schema='p3_layered_card_v1', split=split, id=sample['prediction']['id'], figure=name,
                  source_sample_sha256=digest(OUT3/'samples'/split/(sample['prediction']['id']+'.json')),
                  model_unchanged=True, header_summary_in_image=False, raw_transcript_in_image=False, local_target='original_predicted_class_probability',
                  word_display_rule='Signed maximum-absolute single-position effect; no summation.')
    write_json(OUT3 / 'cards' / (name + '.json'), record)
    return record


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='attachment4'); ap.add_argument('--id', default='19')
    args = ap.parse_args(); apply_style()
    sample = json.loads((OUT3/'samples'/args.split/(args.id+'.json')).read_text())
    record = render_card(args.split, sample)
    print('Rendered', record['figure'], flush=True)
