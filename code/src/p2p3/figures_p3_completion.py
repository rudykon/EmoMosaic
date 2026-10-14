"""Full-position plots and source-media replay from frozen P3 evidence."""
import json
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from project_paths import resolve_asset
from data import ROOT, MODALITIES, write_json
from p3_common import OUT3, digest
from make_cards import frames
from cards_markdown import append_completion

sys.path.insert(0, str(ROOT / 'src/common'))
from plot_style import apply_style, save_figure, heatmap_text_color, COLORS
def copy_figure(problem, name, suffix='.pdf'):
    return str(OUT3 / 'figures' / (name + suffix))

FIG = OUT3 / 'figures'
SIGNED = LinearSegmentedColormap.from_list('p3_signed', ['#0072B2', '#FFFFFF', '#D55E00'])


def load_sample(split, sid):
    return json.loads((OUT3 / 'samples' / split / f'{sid}.json').read_text())


def full_distribution(sample, target):
    """Preserve every eligible position; normalize color within each modality."""
    key = 'class_probability_delta' if target == 'classification' else 'intensity_delta'
    positions = [i for i in range(50) if any(row[i]['eligible'] for row in sample['positions'])]
    values = np.array([[row[i][key] if row[i]['eligible'] else np.nan for i in positions]
                       for row in sample['positions']])
    scales = np.nanmax(np.abs(values), axis=1)
    normalized = values / np.where(scales > 0, scales, 1)[:, None]
    blocks = [np.arange(i, min(i + 20, len(positions))) for i in range(0, len(positions), 20)]
    fig, axes = plt.subplots(len(blocks), 1, figsize=(8.4, 2.15 * len(blocks) + .7), squeeze=False)
    main = sample['prediction']['main_modality_' + target]
    for ax, block in zip(axes[:, 0], blocks):
        matrix = np.ma.masked_invalid(normalized[:, block])
        ax.imshow(matrix, cmap=SIGNED, vmin=-1, vmax=1, aspect='auto', interpolation='nearest')
        labels = []
        for ix in block:
            pos = positions[ix]
            item = sample['positions'][0][pos]
            labels.append(f"{item['token']} [{pos}]")
        ax.set_xticks(range(len(block)), labels, rotation=55, ha='right', fontsize=8)
        ax.set_yticks(range(3), [f'{m}{" *" if m == main else ""}\nmax={v:.2g}'
                                for m, v in zip(MODALITIES, scales)], fontsize=8)
        ax.set_xticks(np.arange(-.5, len(block), 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which='minor', color='#DDDDDD', linewidth=.5)
        ax.tick_params(which='minor', bottom=False, left=False)
    title = 'Original-class probability' if target == 'classification' else 'Fused intensity'
    fig.suptitle(f"Attachment 4 / {sample['prediction']['id']} | {title} | * main modality", fontsize=11)
    fig.text(.5, .012, 'Blue: negative effect   |   White: zero   |   Orange: positive effect; color scaled within each modality',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .95), h_pad=1.8)
    name = f"local_distribution_attachment4_{sample['prediction']['id']}_{target}"
    save_figure(fig, FIG, name)
    plt.close(fig)
    return dict(figure=name, target=target, positions=positions, values=values.tolist(),
                per_modality_max_abs=scales.tolist(), main_modality=main,
                color_normalization='Per modality across all positions; raw maxima printed on row labels.')


def source_replay(sample):
    sid = sample['prediction']['id']
    polarity = {'Negative': '负面', 'Neutral': '中性', 'Positive': '正面'}[
        sample['prediction']['pred_polarity']]
    signal = np.load(OUT3 / 'alignment' / f'{sid}.npz')
    fig = plt.figure(figsize=(8.4, 5.9))
    # 分别设置两处层间留白，为时间轴和三行帧注释保留足够空间。
    grid = fig.add_gridspec(5, 3, height_ratios=[1.05, .26, 1.2, .53, 1.4], hspace=0)
    ax = fig.add_subplot(grid[0, :]); ax.axis('off')
    ax.text(0, 1, '\n'.join(textwrap.wrap(sample['prediction']['raw_text'], 100)),
            transform=ax.transAxes, ha='left', va='top', fontsize=10, linespacing=1.4)
    evidence = sample['top_evidence']['text']
    lines = [f"{r['token']} / {r['word']} [{r['position']}] : {r['class_probability_delta']:+.4f}"
             for r in evidence]
    ax.text(0, .02,
            f'文本证据（词元 / 原词 [位置]；数值为遮挡后{polarity}概率下降量）\n'
            + '   |   '.join(lines[:2]) + '\n' + lines[2],
            transform=ax.transAxes, ha='left', va='bottom', fontsize=8.5)
    ax.set_title('原始转写与文本遮挡效应', loc='left')
    ax = fig.add_subplot(grid[2, :])
    ax.plot(signal['audio_times'], signal['rms'], color='#0072B2', linewidth=.85)
    unique = {}
    for r in sample['top_evidence']['audio']:
        unique.setdefault((r['time_start'], r['time_end']), r)
    top = max(float(signal['rms'].max()), 1e-12)
    for j, ((start, end), r) in enumerate(unique.items()):
        ax.axvspan(start, end, facecolor='#E69F00', alpha=.35, edgecolor='none')
        ax.annotate(f"{r['word']} [{r['position']}]\n{start:.3f}—{end:.3f}秒",
                    xy=((start + end) / 2, top * .25),
                    xytext=(.34 + .24 * j, .92), textcoords='axes fraction',
                    ha='center', va='top', fontsize=8,
                    arrowprops=dict(arrowstyle='-', color='black', lw=.7))
    ax.set_xlim(0, signal['audio_times'][-1]); ax.set_ylim(0, top * 1.55)
    ax.set_xlabel('原音频时间（秒）'); ax.set_ylabel('均方根幅值（仅辅助回看）')
    ax.set_title('音频证据区间；均方根幅值不代表归因分数', loc='left')
    images = frames(sample)
    for j, r in enumerate(sample['top_evidence']['vision']):
        ax = fig.add_subplot(grid[4, j]); ax.axis('off'); frame = r['keyframe']
        ax.imshow(images[frame['frame_index']])
        ax.set_title(f"{r['word']} [{r['position']}] | 原帧{frame['frame_index']}\n"
                     f"呈现时间戳：{frame['pts_relative']:.3f}秒\n"
                     f"{polarity}概率下降量：{r['class_probability_delta']:+.5f}", fontsize=8)
    fig.suptitle(f"附件4 / {sid}号样本 | 预测类别：{polarity} | 情感强度：{sample['prediction']['pred_intensity']:.4f}",
                 fontsize=11, y=.99)
    fig.subplots_adjust(top=.91, bottom=.035, left=.085, right=.98)
    name = f'source_replay_attachment4_{sid}'
    save_figure(fig, FIG, name); plt.close(fig)
    return dict(figure=name, video=sample['video'], source_sha256=digest(resolve_asset(sample['video'])),
                evidence={m: sample['top_evidence'][m] for m in MODALITIES},
                note='CTC candidate timing; raw source frames unchanged. RMS only aids replay.')


def evidence_summary():
    rows = []
    for path in sorted((OUT3 / 'samples/attachment4').glob('*.json')):
        sample = json.loads(path.read_text()); p = sample['prediction']
        row = {k: p[k] for k in ['id', 'pred_polarity', 'pred_intensity', 'main_modality_classification',
                                 'main_modality_intensity']}
        for m in MODALITIES:
            evidence = sample['top_evidence'][m]
            r = evidence[0] if evidence else None
            row.update({f'{m}_word': r['word'] if r else '', f'{m}_token': r['token'] if r else '',
                        f'{m}_position': r['position'] if r else '',
                        f'{m}_time_start': r['time_start'] if r else '',
                        f'{m}_time_end': r['time_end'] if r else '',
                        f'{m}_timing_usable': bool(r and r['timing_usable']),
                        f'{m}_probability_delta': r['class_probability_delta'] if r else '',
                        f'{m}_status': ('estimated' if r['timing_usable'] else 'candidate_low_trust') if r else 'no_observed_evidence'})
            if m == 'vision':
                row['vision_frame'] = r['keyframe']['frame_index'] if r and r['keyframe'] else ''
                row['vision_pts'] = r['keyframe']['pts_relative'] if r and r['keyframe'] else ''
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT3 / 'attachment4_evidence_summary.csv', index=False, encoding='utf-8-sig')
    return rows


def validation_confusion():
    """Figure 6.3: counts and per-class metrics from the same frozen predictions."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    summary = json.loads((OUT3 / 'valid_summary.json').read_text())
    predictions = pd.read_csv(OUT3 / 'validation_predictions_and_explanations.csv')
    with np.load(ROOT / 'outputs/problem2/cache/valid.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(predictions.id.to_numpy(), data['ids'])
        truth = np.sign(data['y']).astype(int) + 1
    probability = predictions[['p_neg', 'p_neu', 'p_pos']].to_numpy()
    assert len(truth) == summary['n'] == 728 and predictions.id.is_unique
    assert np.isfinite(probability).all() and (probability >= 0).all()
    np.testing.assert_allclose(probability.sum(1), 1, atol=1e-7)
    predicted = probability.argmax(1)
    np.testing.assert_array_equal(predictions.pred_polarity,
                                 np.array(['Negative', 'Neutral', 'Positive'])[predicted])
    confusion = confusion_matrix(truth, predicted, labels=[0, 1, 2])
    np.testing.assert_array_equal(confusion, summary['metrics']['confusion'])
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=[0, 1, 2], zero_division=0)
    np.testing.assert_allclose([np.trace(confusion)/len(truth), f1.mean(),
                               np.average(f1, weights=support)],
                              [summary['metrics'][key] for key in
                               ['accuracy', 'macro_f1', 'weighted_f1']])
    labels = ['负向', '中性', '正向']
    with plt.rc_context():
        apply_style()
        plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                             'axes.labelsize': 12, 'xtick.labelsize': 11,
                             'ytick.labelsize': 11, 'legend.fontsize': 11})
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6),
                                 gridspec_kw={'width_ratios': [1, 1.2]})
        ax = axes[0]
        im = ax.imshow(confusion, cmap='Blues', vmin=0, interpolation='nearest')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(confusion[i, j]), ha='center', va='center',
                        color=heatmap_text_color(im, confusion[i, j]), fontsize=17)
        ax.set_xticks(range(3), labels)
        ax.set_yticks(range(3), labels)
        ax.set_xlabel('预测类别'); ax.set_ylabel('真实类别')
        ax.set_title('(a) 三分类混淆矩阵')
        ax = axes[1]
        x = np.arange(3)
        for j, (values, label, color) in enumerate(zip(
                [precision, recall, f1], ['精确率', '召回率', 'F1'], COLORS[:3])):
            bars = ax.bar(x+(j-1)*.25, values*100, width=.23, color=color,
                          edgecolor='black', linewidth=.5, label=label)
            ax.bar_label(bars, labels=[f'{v*100:.1f}' for v in values],
                         padding=3, fontsize=11.5)
        ax.set_xticks(x, [f'{label}\nn={n}' for label, n in zip(labels, support)])
        ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
        ax.set_xlim(-.6, 2.6)
        ax.set_ylabel('指标（%）'); ax.set_xlabel('类别及真实样本数')
        ax.set_title('(b) 各类别识别指标')
        ax.legend()
        fig.tight_layout(w_pad=1.5)
        save_figure(fig, FIG, 'validation_confusion')
    pd.DataFrame(dict(label=labels, support=support, predicted_count=confusion.sum(0),
                      precision=precision, recall=recall, f1=f1)).to_csv(
        OUT3 / 'validation/classification_metrics.csv', index=False, float_format='%.12g')
    for suffix in ['.pdf', '.png']:
        copy_figure(3, 'validation_confusion', suffix)
    return confusion


def main():
    if '--confusion-only' in sys.argv:
        validation_confusion()
        print('Updated Figure 6.3 panels (a)/(b); verified all 728 validation predictions.')
        return
    apply_style(); sample = load_sample('attachment4', '19')
    plots = [full_distribution(sample, target) for target in ['classification', 'intensity']]
    replay = source_replay(sample)
    confusion = validation_confusion()
    rows = evidence_summary()
    write_json(OUT3 / 'paper_checks/paper_completion_assets.json', dict(
        schema='p3_paper_completion_v1', source_sample_sha256=digest(OUT3 / 'samples/attachment4/19.json'),
        distributions=plots, source_replay=replay, confusion=confusion.tolist(), evidence_summary_rows=len(rows)))
    append_completion(OUT3)
    print('Generated full-position distributions, source replay, confusion matrix and 20 evidence summaries.', flush=True)


if __name__ == '__main__':
    main()
