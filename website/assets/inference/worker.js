"use strict";
// This worker has no prediction endpoint or saved-result fallback.
importScripts("vendor/ort/ort.wasm.min.js", "tokenizer.js", "fusion.js");
ort.env.wasm.numThreads = 1; // Pages does not supply cross-origin isolation headers.
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = new URL("vendor/ort/", self.location.href).href;
const sessions = new Map();
let manifest, tokenizer, adapters, zeroAdapter, busy = false;
const fail = message => { throw new Error(message); };
const dispose = values => Object.values(values).forEach(tensor => tensor.dispose());
const int64 = (values, dims) => new ort.Tensor("int64", BigInt64Array.from(values, BigInt), dims);
const bool = values => new ort.Tensor("bool", Uint8Array.from(values), [1, 50]);
const sha = async buffer => [...new Uint8Array(await crypto.subtle.digest("SHA-256", buffer))]
  .map(b => b.toString(16).padStart(2, "0")).join("");

async function asset(path) {
  const meta = manifest.assets[path];
  if (!meta) fail("模型资源清单不完整");
  const base = path.startsWith("models/") ? manifest.assetBaseUrl : self.location.href;
  const url = new URL(path, base);
  url.searchParams.set("sha256", meta.sha256);
  let cache, response;
  try { cache = await caches.open("emomosaic-paper-model-v1"); response = await cache.match(url); } catch (_) { /* Private mode/quota: use network. */ }
  if (!response) response = await fetch(url);
  if (!response.ok) fail("模型下载失败，请检查网络后重试");
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== meta.bytes || await sha(buffer) !== meta.sha256) {
    if (cache) await cache.delete(url).catch(() => {});
    fail("模型文件校验失败，请重试");
  }
  if (cache) await cache.put(url, new Response(buffer)).catch(() => {});
  return buffer;
}

async function session(name, progress) {
  if (sessions.has(name)) return sessions.get(name);
  let bytes;
  if (name === "bert") {
    bytes = new Uint8Array(manifest.encoder.bytes);
    let offset = 0;
    for (let i = 0; i < manifest.encoder.parts.length; i++) {
      progress("加载 BERT 参数", i, manifest.encoder.parts.length);
      const chunk = new Uint8Array(await asset("models/" + manifest.encoder.parts[i]));
      bytes.set(chunk, offset); offset += chunk.length;
    }
    if (offset !== bytes.length || await sha(bytes) !== manifest.encoder.sha256) fail("模型文件校验失败，请重试");
  } else {
    progress("加载 RM-Net 参数");
    bytes = new Uint8Array(await asset("models/original.onnx"));
  }
  progress(name === "bert" ? "初始化 BERT" : "初始化 RM-Net");
  const model = await ort.InferenceSession.create(bytes, {executionProviders: ["wasm"], graphOptimizationLevel: "all"});
  sessions.set(name, model);
  return model;
}

function validateSample(sample) {
  if (!sample || sample.schema !== "emomosaic_aligned_50_v1") fail("输入文件的格式标识不正确");
  if (!Number.isInteger(sample.length) || sample.length < 1 || sample.length > 50) fail("有效长度应为 1 至 50");
  if (!Array.isArray(sample.token_ids) || sample.token_ids.length !== 50 ||
      sample.token_ids.some(id => !Number.isInteger(id) || id < 0 || id >= 30522)) fail("词元编号应为 50 个有效整数");
  if (!Array.isArray(sample.observed) || sample.observed.length !== 3 || sample.observed.some(row =>
    !Array.isArray(row) || row.length !== 50 || row.some((v, i) => typeof v !== "boolean" || (i >= sample.length && v)))) fail("观测掩码应为 3 × 50，且填充位置不可见");
  if (sample.observed[0].some((v, i) => v && sample.token_ids[i] === 0)) fail("文本掩码与词元编号不一致");
  for (const [name, dimension] of [["text", 768], ["audio", 74], ["vision", 35]]) {
    if (!Array.isArray(sample[name]) || sample[name].length !== 50 * dimension ||
        sample[name].some(v => typeof v !== "number" || !Number.isFinite(v) || Math.abs(v) > 1e6)) fail("特征维度或数值不正确");
  }
  return sample;
}

async function predict(input, progress) {
  const start = performance.now();
  let sample, truncated = false;
  if (input.kind === "features") sample = validateSample(input.sample);
  else if (input.kind === "text") {
    if (typeof input.text !== "string" || input.text.length > 10000 || !/[a-z]/i.test(input.text) ||
        /[\p{Script=Han}\p{Script=Hangul}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Cyrillic}]/u.test(input.text)) fail("请输入英文文本，长度不超过 10000 个字符");
  } else fail("输入类型不正确");
  const enabled = input.enabled ?? [true, true, true];
  if (!Array.isArray(enabled) || enabled.length !== 3 || enabled.some(v => typeof v !== "boolean")) fail("模态选择不正确");
  if (!manifest) {
    const response = await fetch("manifest.json", {cache: "no-cache"});
    if (!response.ok) fail("模型资源清单无法加载");
    manifest = await response.json();
    if (manifest.schema !== "emomosaic_browser_inference_v1") fail("模型资源清单不完整");
  }
  if (input.kind === "text") {
    tokenizer ||= BertTokenizer.create(new TextDecoder().decode(await asset("models/vocab.txt")));
    const encoded = tokenizer(input.text.trim());
    truncated = encoded.truncated;
    const observed = encoded.mask.map((v, i) => Boolean(v && encoded.ids[i]));
    const encoder = await session("bert", progress);
    zeroAdapter ||= Object.fromEntries(Object.entries(manifest.adapters[0].fields).map(([name, field]) =>
      [name, new ort.Tensor("float32", new Float32Array(field.length), field.shape)]));
    const feeds = {input_ids: int64(encoded.ids, [1, 50]), attention_mask: bool(observed), keep_mask: bool(observed)};
    progress("编码英文文本");
    let output;
    try {
      output = await encoder.run({...feeds, ...zeroAdapter});
      sample = {length: encoded.length, token_ids: encoded.ids,
        observed: [observed, Array(50).fill(false), Array(50).fill(false)],
        text: Float32Array.from(output.hidden.data), audio: new Float32Array(3700), vision: new Float32Array(1750)};
    } finally { dispose(feeds); if (output) dispose(output); }
  }
  const observed = sample.observed.map((row, m) => row.map(v => v && enabled[m]));
  const coverage = Fusion.fraction(sample.token_ids, observed[0]);
  const coreModel = await session("core", progress);
  progress("计算 RM-Net 五模型集成");
  const feeds = Object.fromEntries([["text", 768], ["audio", 74], ["vision", 35]].map(([name, dim]) =>
    [name, new ort.Tensor("float32", Float32Array.from(sample[name]), [1, 50, dim])]));
  feeds.observed = new ort.Tensor("bool", Uint8Array.from(observed.flat()), [1, 3, 50]);
  feeds.lengths = int64([sample.length], [1]);
  let output, core;
  try {
    output = await coreModel.run(feeds);
    core = {probabilities: Array.from(output.probabilities.data), intensity: output.intensity.data[0],
      weights: Array.from(output.weights.data), contributions: Array.from(output.contributions.data), interaction: output.interaction.data[0]};
  } finally { dispose(feeds); if (output) dispose(output); }
  let expert = null;
  if (coverage > 0) {
    const encoder = await session("bert", progress);
    if (!adapters) {
      progress("加载 LoRA 参数");
      const loaded = [];
      for (const adapter of manifest.adapters) {
        const buffer = await asset("models/" + adapter.file);
        loaded.push(Object.fromEntries(Object.entries(adapter.fields).map(([name, field]) =>
          [name, new ort.Tensor("float32", new Float32Array(buffer, field.offset, field.length), field.shape)])));
      }
      adapters = loaded;
    }
    expert = {probabilities: [0, 0, 0], intensity: 0};
    // Original attention mask is kept; intervention affects expert pooling,
    // as in lora_text_expert.predict_adapter, not raw word deletion.
    const textFeeds = {input_ids: int64(sample.token_ids, [1, 50]), attention_mask: bool(sample.observed[0]), keep_mask: bool(observed[0])};
    try {
      for (let i = 0; i < adapters.length; i++) {
        progress("计算 BERT LoRA 文本专家", i + 1, adapters.length);
        const part = await encoder.run({...textFeeds, ...adapters[i]});
        try {
          part.probabilities.data.forEach((v, j) => { expert.probabilities[j] += v / adapters.length; });
          expert.intensity += part.intensity.data[0] / adapters.length;
        } finally { dispose(part); }
      }
    } finally { dispose(textFeeds); }
  }
  const result = Fusion.combine(core, expert, coverage, manifest.fusionRules);
  if (![...result.probabilities, result.intensity, ...result.contributions, result.interaction, result.loraContribution].every(Number.isFinite) ||
      Math.abs(result.probabilities.reduce((a, b) => a + b, 0) - 1) > 1e-5) fail("模型输出异常，请重新运行");
  return {...result, label: ["Negative", "Neutral", "Positive"][result.probabilities.indexOf(Math.max(...result.probabilities))],
    coreProbabilities: core.probabilities, expertProbabilities: expert?.probabilities ?? null,
    observedCounts: observed.map(row => row.filter(Boolean).length), length: sample.length, truncated,
    model: manifest.model, runtime: manifest.runtime, fusionRulesSha256: manifest.fusionRulesSha256,
    elapsedSeconds: (performance.now() - start) / 1000};
}

self.onmessage = async ({data: {id, input}}) => {
  if (busy) return self.postMessage({id, error: "模型正在运行"});
  busy = true;
  try {
    const result = await predict(input, (phase, current, total) => self.postMessage({id, progress: {phase, current, total}}));
    self.postMessage({id, result});
  } catch (error) {
    self.postMessage({id, error: error.message || "模型无法运行，请使用新版桌面浏览器并重试"});
  } finally { busy = false; }
};
