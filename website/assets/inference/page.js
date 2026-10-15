"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const controls = $("inference-controls"), source = $("inference-source");
  const checkboxes = ["text", "audio", "vision"].map(m => $("inference-" + m + "-enabled"));
  let worker, busy = false, requestId = 0, abort, resultURL, timer;
  const status = text => { $("inference-status").textContent = text; };
  function clearResult() {
    $("inference-result").hidden = true; $("inference-empty").hidden = false;
    $("inference-json").textContent = "";
    if (resultURL) URL.revokeObjectURL(resultURL);
    resultURL = null; $("download-inference").removeAttribute("href");
    $("inference-error").hidden = true;
  }
  function setBusy(value) {
    busy = value; controls.disabled = value;
    $("cancel-inference").hidden = !value;
    $("inference-progress").hidden = !value;
    $("inference-progress").removeAttribute("value");
    if (!value) clearTimeout(timer);
  }
  function stop() {
    requestId++; abort?.abort(); worker?.terminate(); worker = null;
    setBusy(false);
  }
  function error(message) {
    stop(); clearResult(); status("运行未完成");
    $("inference-error").textContent = message;
    $("inference-error").hidden = false;
  }
  function changeSource() {
    clearResult(); status("等待运行");
    $("text-input-panel").hidden = source.value !== "text";
    $("file-input-panel").hidden = source.value !== "file";
    checkboxes.slice(1).forEach(box => { box.disabled = source.value === "text"; });
  }
  function render(result, context) {
    const labels = {Negative: "负向", Neutral: "中性", Positive: "正向"};
    $("inference-label").textContent = labels[result.label];
    $("inference-intensity").textContent = result.intensity.toFixed(4);
    $("inference-probabilities").replaceChildren(...result.probabilities.map((p, i) => {
      const row = document.createElement("div"); row.className = "probability-row";
      if (Object.keys(labels)[i] === result.label) row.classList.add("selected");
      const label = document.createElement("span"); label.textContent = Object.values(labels)[i];
      const track = document.createElement("div"), bar = document.createElement("i");
      bar.style.width = (100 * p) + "%"; track.append(bar);
      const number = document.createElement("strong"); number.textContent = (100 * p).toFixed(2) + "%";
      row.append(label, track, number); return row;
    }));
    $("inference-coverage").textContent = (result.textFraction * 100).toFixed(1) + "%";
    $("inference-class-weight").textContent = (result.expertWeight * 100).toFixed(1) + "%";
    $("inference-reg-weight").textContent = (result.regressionExpertWeight * 100).toFixed(1) + "%";
    $("inference-core").textContent = result.coreIntensity.toFixed(4);
    $("inference-expert").textContent = result.expertIntensity === null ? "未启用" : result.expertIntensity.toFixed(4);
    $("inference-time").textContent = result.elapsedSeconds.toFixed(1) + " s";
    $("inference-context").textContent = `有效长度：${result.length} · 可见位置（文本 / 语音 / 视觉）：${result.observedCounts.join(" / ")}`;
    $("inference-truncation").hidden = !result.truncated;
    const record = JSON.stringify({schema: "emomosaic_browser_prediction_v1", computedAt: new Date().toISOString(),
      inputSource: context, enabled: checkboxes.map(box => box.checked && !box.disabled), ...result}, null, 2);
    $("inference-json").textContent = record;
    resultURL = URL.createObjectURL(new Blob([record + "\n"], {type: "application/json"}));
    $("download-inference").href = resultURL;
    $("inference-result").hidden = false; $("inference-empty").hidden = true;
    status("计算完成");
  }
  async function run(example = false) {
    if (busy) return;
    if (example) { source.value = "example"; checkboxes.forEach(box => { box.checked = true; }); changeSource(); }
    clearResult(); setBusy(true); status("读取输入");
    const id = ++requestId;
    abort = new AbortController();
    // No fixed short timeout: slow laptops may need several minutes. Stop is always available.
    timer = setTimeout(() => error("运行超过 15 分钟，已停止；请检查网络或更换设备后重试"), 15 * 60 * 1000);
    try {
      if (!globalThis.Worker || !globalThis.WebAssembly || !globalThis.crypto?.subtle) throw new Error("需要支持 WebAssembly 的新版浏览器，并通过 HTTPS 或 localhost 访问");
      let input, context;
      if (source.value === "text") {
        input = {kind: "text", text: $("inference-text").value}; context = "English text / local BERT encoding";
      } else {
        let sample;
        if (source.value === "example") {
          const response = await fetch("data/inference/attachment4-19.json", {signal: abort.signal});
          if (!response.ok) throw new Error("案例数据下载失败，请检查网络后重试");
          sample = await response.json(); context = "Attachment 4 / sample 19";
        } else {
          const file = $("inference-file").files[0];
          if (!file) throw new Error("请先选择对齐特征 JSON 文件");
          if (file.size > 3 * 1024 * 1024) throw new Error("输入文件超过 3 MB");
          try { sample = JSON.parse(await file.text()); } catch (_) { throw new Error("输入文件不是有效的 JSON"); }
          context = "Local aligned feature JSON";
        }
        input = {kind: "features", sample};
      }
      if (id !== requestId) return;
      input.enabled = checkboxes.map(box => box.checked && !box.disabled);
      worker ||= new Worker("assets/inference/worker.js");
      worker.onmessage = ({data}) => {
        if (data.id !== requestId) return;
        if (data.error) error(data.error);
        else if (data.result) { setBusy(false); render(data.result, context); }
        else if (data.progress) {
          const {phase, current, total} = data.progress;
          status(phase + (total ? ` · ${current}/${total}` : "…"));
          if (total) { $("inference-progress").max = total; $("inference-progress").value = current; }
          else $("inference-progress").removeAttribute("value");
        }
      };
      worker.onerror = event => { event.preventDefault(); error("模型无法运行，请使用新版桌面浏览器并重试"); };
      worker.postMessage({id, input});
    } catch (e) { if (id === requestId) error(e.message); }
  }
  $("run-example").addEventListener("click", () => run(true));
  $("run-inference").addEventListener("click", () => run());
  $("cancel-inference").addEventListener("click", () => { stop(); clearResult(); status("计算已停止"); });
  $("clear-inference").addEventListener("click", () => {
    stop(); clearResult(); $("inference-text").value = ""; $("inference-file").value = ""; status("输入已清空，模型已释放");
  });
  $("clear-model-cache").addEventListener("click", async () => {
    stop(); clearResult(); setBusy(true);
    try { await caches.delete("emomosaic-paper-model-v1"); status("模型缓存已清除"); }
    catch (_) { status("浏览器未开放模型缓存"); }
    finally { setBusy(false); }
  });
  source.addEventListener("change", changeSource);
  for (const control of [...checkboxes, $("inference-text"), $("inference-file")]) control.addEventListener("input", () => {
    clearResult(); status("输入已更改，请重新计算");
  });
  window.addEventListener("pagehide", stop);
})();
