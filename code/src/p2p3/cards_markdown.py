"""Render explanation indexes from saved results without inference or plotting."""
import argparse
import os
import json
from pathlib import Path
import re
from urllib.parse import quote

MODALITIES = ('text', 'audio', 'vision')
TITLE = '问题三：多模态分层解释卡'
INTRO = (
    '附件4按原始秒数对齐文字、波形、基频、三模态贡献与视频帧；'
    '验证集按官方词元序号排列输入特征与贡献。预测与证据明细列于各图下方。'
    '橙色贡献支持原预测类，蓝色抵消；斜纹和虚线标记低可信候选时间。'
)
COMPLETION_HEADING = '## 19号：完整位置分布与原始素材回溯'


def markdown_text(value):
    """Keep literal tokens and IDs from becoming Markdown or table syntax."""
    text = ' '.join(str(value).splitlines())
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return re.sub(r'([\\`*_{}\[\]#!|$])', r'\\\1', text)


def markdown_table(headers, rows):
    def row(values):
        return '| ' + ' | '.join(markdown_text(value) for value in values) + ' |'
    return '\n'.join([row(headers), row(['---'] * len(headers)),
                      *(row(values) for values in rows)])


def sample_link(split, sid):
    return f'../samples/{quote(str(split), safe="")}/{quote(str(sid), safe="")}.json'


def fusion_card(split, role, sample, figure=None):
    p = sample['prediction']
    sid = p['id']
    figure = figure or f'card_{split}_{sid}'.replace('$', '')
    modal_rows = [
        [m, f"{p['shap_class_' + letter]:+.5f}", f"{p['share_class_' + letter]:.1%}",
         f"{p['shap_intensity_' + letter]:+.5f}", f"{p['share_intensity_' + letter]:.1%}"]
        for m, letter in zip(MODALITIES, 'TAV')
    ]
    evidence_rows = []
    for modality in MODALITIES:
        for r in sample['top_evidence'][modality]:
            interval = (f"{r['time_start']:.3f}–{r['time_end']:.3f} s"
                        if r.get('time_start') is not None else '无原始时间')
            trust = '可用候选' if r.get('timing_usable') else '时间低可信或未提供'
            evidence_rows.append([
                modality, r['token'], r['position'], f"{r['class_probability_delta']:+.5f}",
                f"{r['intensity_delta']:+.5f}", interval, trust,
            ])
    truth = (f"；真值强度 {sample['truth']:.3f}" if 'truth' in sample
             else '；无标签，仅展示预测')
    source = '验证集' if split == 'valid' else 'attachment4'
    return '\n\n'.join([
        f'## {markdown_text(role)} · {markdown_text(sid)}',
        f'样本来源：{source}。',
        f'![{markdown_text(source + "样本 " + sid + " 的分层解释卡")}](../figures/{quote(figure, safe="")}.png)',
        f"{markdown_text(p['pred_polarity'])}；融合强度 {p['pred_intensity']:.4f}{truth}；"
        f"概率间隔 {p['probability_margin']:.4f}。",
        '> ' + markdown_text(p['raw_text']),
        '### 模态作用',
        markdown_table(['模态', '类别Shapley', '类别作用占比', '强度Shapley', '强度作用占比'], modal_rows),
        '### 关键证据',
        markdown_table(['模态', '词元', '位置', '类别概率下降', '强度变化', '候选时间', '时间可信度'], evidence_rows),
        f"分类主模态：{p['main_modality_classification']}；强度主模态：{p['main_modality_intensity']}。"
        f"沿分类排序首次翻转位置数：{p['flip_k']}；未翻转删失：{p['flip_censored']}。"
        '该数量不是全局最小反事实。',
        '遮挡完整融合模型的编码后特征，并重新计算文本权重；类别解释固定为原预测类。'
        '负下降值表示该位置抵消原预测类别。',
        f'[全部位置、八子集、类别翻转路径 JSON]({sample_link(split, sid)})',
    ])


def write_index(out3, cards, title=TITLE, introduction=INTRO):
    path = Path(out3) / 'cards/index.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n\n'.join([f'# {title}', introduction, *cards]) + '\n', encoding='utf-8')
    return path


def append_completion(out3):
    """Refresh the optional supplement once, using already generated assets."""
    out3 = Path(out3)
    path = out3 / 'cards/index.md'
    figures = [
        ('分类目标：完整位置分布', 'local_distribution_attachment4_19_classification'),
        ('强度目标：完整位置分布', 'local_distribution_attachment4_19_intensity'),
        ('原始素材回溯', 'source_replay_attachment4_19'),
    ]
    sources = [out3 / 'samples/attachment4/19.json', out3 / 'attachment4_evidence_summary.csv',
               *(out3 / 'figures' / (name + '.png') for _, name in figures)]
    if not path.exists() or not all(p.is_file() for p in sources):
        return False
    sample = json.loads(sources[0].read_text(encoding='utf-8'))
    p = sample['prediction']
    names = {'text': '文本', 'audio': '语音', 'vision': '视觉'}
    positions = {r['position'] for modality in sample['positions'] for r in modality if r['eligible']}
    supplement = [
        COMPLETION_HEADING,
        f"分类主模态为{names[p['main_modality_classification']]}，"
        f"强度主模态为{names[p['main_modality_intensity']]}。"
        f'两张分布图覆盖全部{len(positions)}个有效内容位置；颜色按各模态内部尺度归一，'
        '原始最大效应写在行标签中。',
        *(f'![{label}](../figures/{name}.png)' for label, name in figures),
        '原始帧与RMS仅辅助素材回看，归因来自官方特征的遮挡。候选时间通过项目阈值，仍非人工真值。',
        '[全部20条证据定位汇总 CSV](../attachment4_evidence_summary.csv)',
    ]
    content = path.read_text(encoding='utf-8').split('\n' + COMPLETION_HEADING)[0].rstrip()
    path.write_text(content + '\n\n' + '\n\n'.join(supplement) + '\n', encoding='utf-8')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(os.environ.get('COLING_P3_OUTPUT_ROOT', Path(__file__).resolve().parents[2] / 'outputs/problem3')))
    args = parser.parse_args()
    selection = json.loads((args.output_dir / 'cards/selection.json').read_text(encoding='utf-8'))['samples']
    cards = []
    for row in selection:
        sample = json.loads((args.output_dir / 'samples' / row['split'] / (row['id'] + '.json')).read_text(encoding='utf-8'))
        cards.append(fusion_card(row['split'], row['role'], sample, row.get('figure')))
    path = write_index(args.output_dir, cards)
    append_completion(args.output_dir)
    print(f'Wrote {len(cards)} explanation cards to {path}')


if __name__ == '__main__':
    main()
