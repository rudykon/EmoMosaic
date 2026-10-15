'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const escapeHTML = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const signed = value => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(4)}`;
const math = source => EmoMath.inline(source);
const mathTerm = value => value < 0 ? `(${value.toFixed(4)})` : value.toFixed(4);
const modalityNames = {text:'文本', audio:'语音', vision:'视觉'};

// Shared navigation, preferences and keyboard-accessible search.
const themeButton = $('#theme-toggle');
function updateThemeLabel() {
  const dark = document.documentElement.dataset.theme === 'dark';
  themeButton.setAttribute('aria-label', dark ? '切换至浅色模式' : '切换至深色模式');
  themeButton.title = themeButton.getAttribute('aria-label');
}
updateThemeLabel();
themeButton.addEventListener('click', () => {
  const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('emomosaic-theme', theme); } catch (_) { /* Preference still works for this page. */ }
  updateThemeLabel();
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
  let saved;
  try { saved = localStorage.getItem('emomosaic-theme'); } catch (_) { /* Use system preference. */ }
  if (!saved) {
    document.documentElement.dataset.theme = event.matches ? 'dark' : 'light';
    updateThemeLabel();
  }
});
$('#menu-toggle').addEventListener('click', () => {
  const open = $('#main-nav').classList.toggle('open');
  $('#menu-toggle').setAttribute('aria-expanded', String(open));
  $('#menu-toggle').setAttribute('aria-label', open ? '关闭导航' : '打开导航');
});
const backTop = $('#back-top');
window.addEventListener('scroll', () => { backTop.hidden = window.scrollY < 550; }, {passive:true});
backTop.addEventListener('click', () => {
  window.scrollTo({top:0, behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
  $('#main').focus({preventScroll:true});
});
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) if (entry.isIntersecting) {
      $$('.toc a').forEach(link => link.classList.toggle('active', link.hash === `#${entry.target.id}`));
    }
  }, {rootMargin:'-15% 0px -60% 0px'});
  $$('main section[id]').forEach(section => observer.observe(section));
}

const searchIndex = [
  ['研究概述', 'RM + BERT LoRA · 多模态情感预测', 'index.html', 'CMU-MOSEI overview'],
  ["特征对齐", "特征提取与时序对齐", "problem1.html", "CTC 质量掩码"],
  ["模型方法", "缺失鲁棒预测", "method.html", "RM LoRA 双视图训练"],
  ["预测解释", "预测解释", "problem3.html", "Shapley 局部证据"],
  ["双视图训练", "渐进式片段遮挡、重建与一致性约束", "method.html#training", "连续缺失 训练"],
  ['研究任务', '特征构建、缺失预测与模型解释', 'index.html#research', '问题1 问题2 问题3'],
  ['特征提取与时序对齐', '文字 768 维、语音 74 维、视觉 35 维', 'problem1.html#alignment', 'aligned_50 特征提取'],
  ['模型结构', '五种子 RM 与五个 BERT LoRA 文本专家', 'method.html#architecture', 'Transformer 可靠性门控'],
  ['文本可用度约束融合', '按文本可见比例调整分类与回归权重', 'method.html#fusion', '0.8 0.6 偏置 f²'],
  ['双目标解释', '强度与原预测类概率的 Shapley、位置遮挡', 'method.html#explanation', '归因 保真度'],
  ['总体性能', '四组冻结结果、混淆矩阵、视频簇区间', 'results.html#evaluation', '准确率 Macro-F1 MAE Pearson'],
  ['候选比较', 'RM、LoRA、BERT、Qwen 和 SVD 候选', 'results.html#comparison', 'baseline ablation 消融 基线'],
  ['解释保真度', '删除、保留与真实删词对照', 'results.html#evidence-results', 'AUC 遮挡'],
  ['评估设置', '3395 / 728 / 727，历史测试使用与缺失条件', 'results.html#protocol', '数据划分'],
  ['样本分析', '20 条样本、8 个模态子集的冻结预测', 'explore.html#playground', '模态 开关 demo'],
  ['模态归因', '类别概率、情感强度、RM 门控和 LoRA 贡献', 'explore.html#attribution', 'Shapley'],
  ['局部证据', '词元、原词、遮挡变化与候选时间', 'explore.html#evidence', '关键词 解释'],
  ['在线推理', '本机运行论文模型与真实数据案例', 'inference.html', 'inference CPU WASM ONNX 一键测试'],
  ['离线应用', '应用功能与离线验证截图', 'app.html#interface', 'UI 手机 录音 视频'],
  ['端侧运行', '共享 BERT 与五个 LoRA，手机音视频迁移试验', 'app.html#deployment', 'Android 离线 WASM APK'],
  ['复现说明', '固定基座、RM 权重、LoRA 和推理入口', 'reproduce.html#start', '下载 依赖 Python CPU'],
  ['研究资料', '源码、模型权重、数据与网站快照', 'reproduce.html#resources', 'GitHub Hugging Face 下载 数据集 模型权重'],
];
const searchDialog = $('#search-dialog');
const searchInput = $('#search-input');
function renderSearch() {
  const query = searchInput.value.trim().toLocaleLowerCase();
  const terms = query.split(/\s+/).filter(Boolean);
  const matches = searchIndex.filter(item => terms.every(term => (item.join(' ') + ' ' + EmoI18n.english(item.join(' '))).toLocaleLowerCase().includes(term)));
  $('#search-count').textContent = query ? `${matches.length} 条结果` : '项目内容';
  $('#search-results').innerHTML = matches.length ? matches.map(([title, description, url]) =>
    `<a class="search-result" href="${url}"><strong>${escapeHTML(title)} →</strong><span>${escapeHTML(description)}</span></a>`
  ).join('') : '<p class="empty-state">无匹配结果</p>';
}
function openSearch() {
  if (!searchDialog.open) searchDialog.showModal();
  renderSearch();
  searchInput.focus();
}
$('.search-trigger').addEventListener('click', openSearch);
$('.close-search').addEventListener('click', () => searchDialog.close());
searchInput.addEventListener('input', renderSearch);
searchDialog.addEventListener('click', event => {
  if (event.target === searchDialog) {
    const rect = searchDialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) searchDialog.close();
  }
});
$('#search-results').addEventListener('click', event => { if (event.target.closest('a')) searchDialog.close(); });
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && searchDialog.open) {
    event.preventDefault();
    searchDialog.close();
    return;
  }
  const editing = /INPUT|TEXTAREA|SELECT/.test(event.target.tagName) || event.target.isContentEditable;
  if ((event.key === '/' && !editing) || ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k')) {
    event.preventDefault(); openSearch();
  }
});

$$('.copy-code').forEach(button => {
  button.addEventListener('click', async () => {
    const code = $('code', button.closest('.code-block'));
    try {
      await navigator.clipboard.writeText(code.textContent);
      button.textContent = '已复制 ✓';
    } catch (_) {
      const range = document.createRange(); range.selectNodeContents(code);
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      button.textContent = '已选中，请复制';
    }
    setTimeout(() => { button.textContent = '复制代码'; }, 2500);
  });
});

const classNames = ['负向', '中性', '正向'];
const className = label => ({Negative:'负向', Neutral:'中性', Positive:'正向'})[label];
const predictedClass = probabilities => probabilities.indexOf(Math.max(...probabilities));

function initResults(data) {
  const selection = () => ({split:$('input[name=split]:checked').value, condition:$('input[name=condition]:checked').value});
  function renderMetrics() {
    const {split, condition} = selection();
    const metrics = data.metrics[split][condition];
    const cards = [['Macro-F1','macro_f1','三类宏平均 F1 ↑'],['Accuracy','accuracy','三分类准确率 ↑'],['Weighted-F1','weighted_f1','按类别支持数加权 ↑'],['MAE','mae','强度平均绝对误差 ↓'],['Pearson r','corr','强度相关系数 ↑']];
    $('#metrics').innerHTML = cards.map(([title,key,note]) => `<div class="metric-card"><span>${title}</span><strong>${metrics[key].toFixed(4)}</strong><small>${note}</small></div>`).join('');
    $('#metric-note').textContent = `${split === 'test' ? '历史留出结果，测试集曾被查看。' : '验证集曾参与模型选择。'}Macro-F1 的视频簇 95% 区间 [${metrics.f1_ci95.map(v=>v.toFixed(4)).join(', ')}]。${condition === 'clean' ? '清洁输入含自然无信息帧。' : '40% 编码后位置遮挡。'}`;
    const max = Math.max(...metrics.confusion.flat());
    $('#confusion').innerHTML = `<table class="confusion-table"><caption class="sr-only">混淆矩阵：行是真实类别，列是预测类别</caption><thead><tr><th scope="col">真实 ↓ / 预测 →</th>${classNames.map(label=>`<th scope="col">${label}</th>`).join('')}</tr></thead><tbody>${metrics.confusion.map((row,index)=>`<tr><th scope="row">${classNames[index]}</th>${row.map(value=>`<td style="--heat:${(value/max*32).toFixed(1)}%">${value}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
    renderComparison();
  }
  function renderComparison() {
    const {split, condition} = selection();
    const key = $('#comparison-metric').value;
    const rows = data.comparison.map(row=>({...row, values:row.metrics[split][condition]}));
    const lower = Math.min(0,...rows.map(row=>row.values[key]));
    const upper = Math.max(1,...rows.map(row=>row.values[key]));
    const range = upper-lower;
    $('#comparison-chart').innerHTML = rows.map(row=>`<div class="bar-row ${row.model===data.provenance.model?'featured':''}"><span class="bar-label">${escapeHTML(row.name)}</span><div class="bar-track"><div class="bar-fill" style="left:${(Math.min(0,row.values[key])-lower)/range*100}%;width:${Math.abs(row.values[key])/range*100}%"></div></div><strong>${row.values[key].toFixed(4)}</strong></div>`).join('')+`<div class="chart-axis"><span>${lower.toFixed(1)}</span><span>${((upper+lower)/2).toFixed(1)}</span><span>${upper.toFixed(1)}</span></div>`;
    $('#comparison-table').innerHTML = `<table><caption>${split==='test'?'测试集':'验证集'} · ${condition==='clean'?'清洁输入':'40% 混合缺失'}</caption><thead><tr><th scope="col">候选</th><th scope="col">Macro-F1</th><th scope="col">Accuracy</th><th scope="col">MAE</th><th scope="col">Pearson r</th></tr></thead><tbody>${rows.map(row=>`<tr><th scope="row">${escapeHTML(row.name)}</th>${['macro_f1','accuracy','mae','corr'].map(k=>`<td>${row.values[k].toFixed(4)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  }
  const stats=data.test_mean_f1;
  $('#mean-statistics').innerHTML=`测试集两视图平均 ${math(`\\operatorname{Macro\\text{-}F1} = ${stats.new.toFixed(6)}`)}，相对原 RM 辅助头提高 ${stats.delta.toFixed(6)}；差值的 95% 视频簇区间 ${math(`[${stats.delta_ci95.map(v=>v.toFixed(6)).join(', ')}]`)}，双侧簇置换 ${math(`p = ${stats.cluster_permutation_p_two_sided.toFixed(5)}`)}。`;
  const fidelityRows = [
    ['intensity','delete','强度 / 删除','绝对变化，越大影响越强'],
    ['intensity','retain','强度 / 保留','越接近零，越接近原强度'],
    ['classification','delete','概率 / 删除','有符号下降，越大影响越强'],
    ['classification','retain','概率 / 保留','下降越小，原预测类概率越高'],
  ];
  $('#fidelity-table').innerHTML = fidelityRows.map(([target,operation,label,note])=>{
    const row=data.fidelity[target][operation];
    return `<tr><th scope="row">${label}</th><td>${row.selected.mean.toFixed(4)}</td><td>${row.random.mean.toFixed(4)}</td><td>${note}</td></tr>`;
  }).join('');
  $$('input[name=split],input[name=condition]').forEach(input=>input.addEventListener('change',renderMetrics));
  $('#comparison-metric').addEventListener('change',renderComparison);
  renderMetrics();
}

function initExplorer(data) {
  const samples = data.samples;
  const sampleSelect = $('#sample-select');
  const toggles = $$('.modality-toggle');
  let current = null;
  function renderPrediction() {
    if (!current) return;
    const subset = toggles.reduce((mask,input)=>mask | (input.checked ? Number(input.value) : 0),0);
    const {intensity:value,probabilities} = current.subsets[String(subset)];
    const predicted = predictedClass(probabilities);
    $('#prediction-value').textContent = signed(value);
    $('#prediction-polarity').textContent = classNames[predicted];
    $('#prediction-polarity').dataset.class = data.labels[predicted];
    $('#sentiment-marker').style.left = `${Math.max(0,Math.min(100,(value+3)/6*100))}%`;
    $('#full-prediction').textContent = `${signed(current.prediction)} · ${className(current.polarity)}`;
    $('#prediction-delta').textContent = `强度变化 ${signed(value-current.prediction)}`;
    $('#class-probabilities').innerHTML=probabilities.map((p,i)=>`<div class="probability-row ${i===predicted?'selected':''}"><span>${classNames[i]}</span><div><i style="width:${p*100}%"></i></div><strong>${(p*100).toFixed(2)}%</strong></div>`).join('');
    const names = ['文本','语音','视觉'].filter((_,index)=>subset & (1<<index));
    $('#subset-note').textContent = `${subset ? names.join(' + ') : '全空输入 · 先验与固定类别偏置'}。类别取概率最大项，不由强度阈值推断。`;
  }
  function renderAttribution() {
    if (!current) return;
    const kind = $('#attribution-type').value;
    const classification = $('#explanation-target').value==='classification';
    const isGate = kind === 'gate';
    const key = kind==='shapley' ? (classification?'class_shapley':'shapley') : kind;
    const entries = current.modalities.map(item=>({name:modalityNames[item.modality],value:item[key],modality:item.modality}));
    if(kind==='contribution') entries.push({name:'LoRA 专家',value:current.expert_contribution,modality:'expert'},{name:'RM 交互项',value:current.interaction,modality:'interaction'});
    const max = isGate ? 1 : Math.max(...entries.map(item=>Math.abs(item.value)),0.0001)*1.1;
    $('#attribution-chart').innerHTML = entries.map(item=>{
      const width = Math.abs(item.value)/max*(isGate?100:50);
      const left = isGate ? 0 : (item.value>=0 ? 50 : 50-width);
      return `<div class="attribution-row" data-modality="${item.modality}"><span>${item.name}</span><div class="attribution-track ${isGate?'unsigned':''}"><div class="attribution-fill" style="width:${width}%;left:${left}%"></div></div><strong>${isGate?(item.value*100).toFixed(1)+'%':signed(item.value)}</strong></div>`;
    }).join('');
    const baseline = classification?current.subsets['0'].probabilities[current.target_class]:current.subsets['0'].intensity;
    const target = classification?current.probabilities[current.target_class]:current.prediction;
    const notes = {
      shapley:classification?`目标固定为完整输入的「${className(current.polarity)}」类概率，正值表示相对空输入提高该类概率。`:'目标为最终融合情感强度，正值表示相对空输入提高强度。',
      gate:'RM 核心内部门控，不包含 LoRA 专家；不是完整模型的类别归因。',
      contribution:'融合强度的可加分解：缩放后的 RM 三模态贡献、RM 交互项与 LoRA 专家贡献。',
    };
    $('#attribution-note').textContent = `${notes[kind]} 均解释完整输入，不随上方模态开关变化。`;
    $('#attribution-equation').innerHTML = isGate
      ? `RM 内部门控之和：${math(`\\sum_{m\\in\\{\\mathrm{T},\\mathrm{A},\\mathrm{V}\\}} w_m = ${(entries.reduce((sum,item)=>sum+item.value,0)*100).toFixed(1)}\\%`)}`
      : kind==='shapley' ? `空输入基准：${math(`F(\\varnothing) = ${baseline.toFixed(4)}`)}<br>${math(`${entries.map(item=>mathTerm(item.value)).join(' + ')} + ${mathTerm(baseline)} \\approx ${target.toFixed(4)}`)}`
      : math(`${entries.map(item=>mathTerm(item.value)).join(' + ')} \\approx ${current.prediction.toFixed(4)}`);
  }
  function renderEvidence() {
    if(!current)return;
    const target=$('#explanation-target').value;
    const classification=target==='classification';
    $('#evidence-note').innerHTML=classification?`完整输入「${className(current.polarity)}」类的前三个文本位置。${math('\\Delta')} 为该类概率下降；负值表示该位置抵消原类别。`:`完整输入按绝对强度变化排序的前三个文本位置。${math('\\Delta')} 保留方向；词元效应包含上下文与融合权重变化。`;
    const evidence=current.evidence[target];
    $('#evidence-list').innerHTML=evidence.length?evidence.map(item=>`<article class="evidence-item"><span class="mini-tag">词元位置 ${item.position}</span><strong>${escapeHTML(item.token)}</strong><span>${math(`\\Delta = ${(classification?item.class_probability_delta:item.intensity_delta).toFixed(4)}`)}</span><small>原词：${escapeHTML(item.word||item.token)}<br>字符区间 [${item.char_start}, ${item.char_end})<br>${item.timing_usable?`候选时间 ${item.time_start.toFixed(2)}–${item.time_end.toFixed(2)} s`:'时间定位低可信 / 不可用'}</small></article>`).join(''):'<p class="empty-state">该样本没有可展示的文本位置。</p>';
  }
  function renderSample() {
    current = samples.find(sample=>sample.id===sampleSelect.value);
    if (!current) return;
    $('#sample-text').textContent = current.text;
    const quality = current.id==='15'||current.id==='16' ? ' · 时间映射低可信' : (['01','13','18'].includes(current.id)?' · 人脸检出缺失或很低':'');
    $('#sample-id').textContent = `样本 ${current.id} · 完整输入预测：${className(current.polarity)}${quality}`;
    renderPrediction(); renderAttribution();renderEvidence();
  }
  function filterSamples() {
    const filter = $('#sample-filter').value;
    const filtered = samples.filter(sample=>filter==='all'||sample.polarity===filter);
    const previous = sampleSelect.value;
    sampleSelect.innerHTML = filtered.map(sample=>`<option value="${sample.id}">${sample.id} · ${className(sample.polarity)}</option>`).join('');
    const empty = filtered.length === 0;
    [sampleSelect,$('#next-sample'),$('#reset-modalities'),...toggles].forEach(input=>{input.disabled=empty;});
    $('#sample-counter').textContent = `${filtered.length} / ${samples.length} 条`;
    $('#explorer-error').hidden = !empty;
    $('#explorer-error').textContent = empty ? '该极性下没有样本，请选择其他筛选条件。' : '';
    if (!empty) {
      sampleSelect.value = filtered.some(sample=>sample.id===previous) ? previous : (filtered.some(sample=>sample.id==='19')?'19':filtered[0].id);
      renderSample();
    } else {
      current = null;
      $('#sample-text').textContent = '当前筛选没有样本。';
      $('#sample-id').textContent = '附件 4';
      ['#prediction-value','#prediction-polarity','#full-prediction'].forEach(s=>{$(s).textContent='—';});
      ['#prediction-delta','#subset-note','#attribution-note','#evidence-note','#class-probabilities','#attribution-chart','#attribution-equation','#evidence-list'].forEach(s=>{$(s).textContent='';});
      $('#sentiment-marker').style.left = '50%';
    }
  }
  sampleSelect.addEventListener('change',renderSample);
  $('#sample-filter').addEventListener('change',filterSamples);
  $('#next-sample').addEventListener('click',()=>{sampleSelect.selectedIndex=(sampleSelect.selectedIndex+1)%sampleSelect.options.length;renderSample();});
  toggles.forEach(input=>input.addEventListener('change',renderPrediction));
  $('#reset-modalities').addEventListener('click',()=>{toggles.forEach(input=>{input.checked=true;});renderPrediction();});
  $('#attribution-type').addEventListener('change',renderAttribution);
  $('#explanation-target').addEventListener('change',()=>{renderAttribution();renderEvidence();});
  filterSamples();
}

function initMethod(data){
  const slider=$('#visibility');
  function render(){
    const f=Number(slider.value)/100;
    $('#visibility-value').textContent=`${slider.value}%`;
    $('#classification-weight').textContent=`${(data.rules.classification_w0*f**data.rules.classification_k*100).toFixed(1)}%`;
    $('#regression-weight').textContent=`${(data.rules.regression_w0*f**data.rules.regression_k*100).toFixed(1)}%`;
  }
  slider.disabled=false;
  slider.addEventListener('input',render);render();
}
const page = document.body.dataset.page;
if (['results','explore','method'].includes(page)) {
  fetch('data/results.json')
    .then(response=>{if(!response.ok)throw new Error(`HTTP ${response.status}`);return response.json();})
    .then(data=>{if(data.schema!=='emomosaic_website_fusion_v2')throw new Error('Unexpected data version');if(page==='results')initResults(data);else if(page==='explore')initExplorer(data);else initMethod(data);})
    .catch(()=>{
      const target = page==='results' ? $('#metrics') : page==='explore'?$('#explorer-error'):$('.fusion-simulator');
      target.hidden=false;
      target.innerHTML='<div class="empty-state">数据加载失败，请刷新重试。<br><a href="reproduce.html#resources">研究报告 →</a></div>';
      if(page==='results') $$('input[name=split],input[name=condition],#comparison-metric').forEach(input=>{input.disabled=true;});
      if(page==='explore'){
        $('#sample-text').textContent='样本数据未加载。';
        $('#sample-select').innerHTML='<option>数据未加载</option>';
        $$('#sample-filter,#sample-select,#next-sample,#reset-modalities,#attribution-type,#explanation-target,.modality-toggle').forEach(input=>{input.disabled=true;});
      }
    });
}

document.addEventListener('languagechange', () => { if (searchDialog.open) renderSearch(); });
