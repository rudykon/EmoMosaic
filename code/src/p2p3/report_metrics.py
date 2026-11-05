"""Read historical RM report metrics from their original experiment artifacts."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCHEME_NOTE = (
    '本报告保留原 RM-Net 实验记录。2026-09-26 曾采用五种子辅助分类头分别 softmax 后平均概率取最大类，'
    '强度取原 RM 回归均值；该方案现为历史基线。当前最终方案为 RM + BERT LoRA，'
    '默认预测入口、附件3及问题3的双目标解释均已迁移，定义和实测指标见 docs/final_method.md。'
    '以下辅助头分类、原 RM 强度和旧阈值消融分别按原实验口径报告；旧 margin、κ、类别翻转和错误分层'
    '不能当作辅助头或最终融合分类解释。历史 test 已被查看，既有分数不构成新的独立确认。'
)


def load_legacy_auxiliary_metrics(root=ROOT):
    """Combine probability-mean auxiliary classification with RM regression.

    The ensemble artifact's classification uses the historical intensity
    thresholds. Only its regression fields belong in this auxiliary baseline.
    """
    out = Path(root) / 'outputs/problem2'
    evaluation = json.loads(
        (out / 'boosted_classifier/20260925_225440/evaluation.json').read_text(encoding='utf-8')
    )
    regression = json.loads((out / 'ensemble_metrics.json').read_text(encoding='utf-8'))
    return {
        split: {
            condition: {
                **evaluation['splits'][split][condition]['rmnet_auxiliary_saved_baseline'],
                **{key: regression[split][condition][key]
                   for key in ('mae', 'corr', 'corr_defined')},
            }
            for condition in ('clean', 'missing')
        }
        for split in ('valid', 'test')
    }
