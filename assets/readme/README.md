# README assets

These selected assets support the [English README](../../README.md) and [Chinese README](../../README.zh-CN.md). The manuscript itself is not bundled.

The method overview is rendered from Figure 1 of the current English manuscript, maintained locally as `paper/figures/fig_01_model_overview.pdf`. Only the PNG export is included here.

The logos and the two historical result/explanation figures are unmodified copies from [rudykon/EmoMosaic_GMCM2026](https://github.com/rudykon/EmoMosaic_GMCM2026/blob/50202237f5f38f1e44ee916c596d9ab57a92d94e) at commit `50202237f5f38f1e44ee916c596d9ab57a92d94e`. Those two figures retain their original Chinese labels; the English README supplies English captions and alternative text.

| Local asset | Original source |
| --- | --- |
| [emomosaic-horizontal.svg](emomosaic-horizontal.svg) | [Light logo](https://github.com/rudykon/EmoMosaic_GMCM2026/blob/50202237f5f38f1e44ee916c596d9ab57a92d94e/docs/EmoMosaic_logo/emomosaic-horizontal.svg) |
| [emomosaic-horizontal-dark.svg](emomosaic-horizontal-dark.svg) | [Dark logo](https://github.com/rudykon/EmoMosaic_GMCM2026/blob/50202237f5f38f1e44ee916c596d9ab57a92d94e/docs/EmoMosaic_logo/emomosaic-horizontal-dark.svg) |
| [method.png](method.png) | Current manuscript Figure 1: parallel predictors, availability-gated fusion and post-hoc explanations; exported from local `paper/figures/fig_01_model_overview.pdf` |
| [missingness-gains.png](missingness-gains.png) | [Figure 5.7: fusion gains under missingness](https://github.com/rudykon/EmoMosaic_GMCM2026/blob/50202237f5f38f1e44ee916c596d9ab57a92d94e/paper/figures/ch05_problem2/fig_05_07_final_missing_grid_gains.png) |
| [explanation-sample19.png](explanation-sample19.png) | [Figure 6.13: Attachment 4 sample 19](https://github.com/rudykon/EmoMosaic_GMCM2026/blob/50202237f5f38f1e44ee916c596d9ab57a92d94e/paper/figures/ch06_problem3/fig_06_13_attachment4_19_modality_card.png) |

## Method figure export

The one-page source PDF is rendered without content changes as a 3200 × 1600 PNG. From the research workspace, where the source PDF is available:

```bash
pdftoppm -f 1 -singlefile -scale-to 3200 -png \
  paper/figures/fig_01_model_overview.pdf assets/readme/method
```

| File | SHA-256 |
| --- | --- |
| Source PDF | `81421b7a9fb8d723ffcff8391e33653064ed496230a7da8c100e447feaffc74e` |
| PNG export | `262eb3ede55bc2c68bd48675ab83e9a77c3d594bcda019c0b242b770526c9fce` |

## Reported results

The result tables quote the `original_rm_lora_fusion_bias` rows of the [historical evaluation CSV](https://github.com/rudykon/EmoMosaic_GMCM2026/blob/50202237f5f38f1e44ee916c596d9ab57a92d94e/outputs/problem2/improvement/final_evaluation/headline_metrics.csv), rounded to four decimal places. The mean test Macro-F1 comparison uses the arithmetic mean of the `clean` and `mixed40` views for `original_rm_lora_fusion_bias` and `original_rm`.

Historical evaluation does not constitute a new blind test. These assets are documentation excerpts; downloading them does not restore model weights, input data or experiment caches. Original code licensing does not relicense third-party data or media depicted in the figures; see the [dataset terms](https://huggingface.co/datasets/config-h/EmoMosaic/blob/main/TERMS.md).
