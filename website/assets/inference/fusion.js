"use strict";
// Frozen paper fusion rule. Each branch averages probabilities over five seeds.
const Fusion = (() => {
  function fraction(ids, observed) {
    let content = 0, informative = 0;
    ids.forEach((id, i) => {
      if (id !== 0 && id !== 101 && id !== 102) {
        content++;
        if (id !== 100 && observed[i]) informative++;
      }
    });
    return informative / Math.max(content, 1);
  }
  function combine(core, expert, coverage, rules) {
    const wc = coverage > 0 ? rules.fusion.w0 * coverage ** rules.fusion.k : 0;
    const wr = coverage > 0 ? rules.regression_w0 * coverage ** rules.regression_k : 0;
    if ((wc || wr) && !expert) throw new Error("LoRA 专家输出缺失，请重新运行");
    const scores = core.probabilities.map((p, i) =>
      Math.log(Math.max(1e-12, Math.min(1, (1 - wc) * p + wc * (expert?.probabilities[i] ?? 0)))) + rules.fusion.bias[i]);
    const max = Math.max(...scores), exp = scores.map(v => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return {probabilities: exp.map(v => v / sum), intensity: (1 - wr) * core.intensity + wr * (expert?.intensity ?? 0),
      coreIntensity: core.intensity, expertIntensity: expert?.intensity ?? null,
      expertWeight: wc, regressionExpertWeight: wr, textFraction: coverage,
      contributions: core.contributions.map(v => (1 - wr) * v),
      interaction: (1 - wr) * core.interaction, loraContribution: wr * (expert?.intensity ?? 0),
      explanationTarget: "final_intensity_algebraic_components", coreWeights: core.weights};
  }
  return { fraction, combine };
})();
if (typeof module !== "undefined") module.exports = Fusion;
