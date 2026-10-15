# EmoMosaic 项目网页

沿用 [原 EmoMosaic 网站](https://rudykon.github.io/EmoMosaic_GMCM2026/) 的九页结构、Logo、CSS、导航、双语切换、搜索与交互。当前站点地址配置为 `https://rudykon.github.io/EmoMosaic/`。

## 本地构建与预览

构建和静态核验只需要 Python 3.11+ 标准库，无需安装模型、训练依赖或 Node 包：

```bash
python3 scripts/build_pages.py
python3 scripts/check_pages.py
python3 scripts/sync_site_i18n.py --check
python3 -m http.server 8000 --directory _site --bind 127.0.0.1
```

访问 `http://127.0.0.1:8000/`。使用 HTTP 预览，直接双击 HTML 文件会受到浏览器对 `fetch` 和 Worker 的限制。部署到 GitHub Pages 后使用 HTTPS。

## GitHub Pages 部署

1. 将 `website/`、`scripts/build_pages.py`、`scripts/check_pages.py`、`scripts/sync_site_i18n.py` 和 `.github/workflows/pages.yml` 提交到 `rudykon/EmoMosaic` 的 `main` 分支。
2. 在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。
3. 推送网页变更，或在 **Actions → Deploy EmoMosaic website → Run workflow** 手动运行。

工作流只打包 `_site/`，不会上传整个研究工作区。部署成功后访问 `https://rudykon.github.io/EmoMosaic/`。构建产物 `_site/` 不需要提交。

当前公共仓库的 `.gitignore` 会忽略所有名为 `paper/` 的目录；首次迁移时请加入以下例外，以保留网站的图件、第三方说明等静态资源：

```gitignore
/_site/
!/website/
!/website/**
/website/assets/inference/models/
```

源文件所在目录本身不要求是 Git 仓库；可以复制以上文件到已有仓库 checkout 后提交。无需初始化或覆盖本地研究目录。

## 内容与资源

- `template.html`：共享页眉、主导航、侧栏、页脚、搜索和元信息。
- `pages/`：九个页面的中文源内容。
- `assets/style.css`：沿用原站样式，包括移动端和深色主题，并补充论文原图的首页展示样式。
- `i18n/en.json`：英文词典；构建自动生成 `assets/i18n-en.js`。
- `data/results.json`：冻结的实验与样本快照。样本页回放 20 × 8 组真实保存结果；在线推理页执行实际模型计算，两者分别标注。
- `assets/paper/`：当前论文图 1–12 的原始 PDF 与高清 PNG 预览；来源路径、图号、PDF 和预览哈希记录在 `sources.json`。
- `assets/app/`：历史应用截图，用于应用界面展示。
- `assets/inference/`：浏览器推理代码、ONNX Runtime、模型清单、许可证。
- `data/browser-inference-validation.json`、`data/app-verification.json`：原站历史核验记录；它们不是本次迁移测试的结果。

网页源码、数据和 Python 模型入口分别指向当前 GitHub 仓库及既有 Hugging Face 发布。复现页命令依据当前公开仓库 README；本地研究工作区可能包含更多未发布材料。

## 浏览器模型依赖

首次完整运行约下载 254 MB，仅在访问者主动点击运行后加载。33 项模型/词表资源复用原站的公开静态地址，基址位于 `assets/inference/manifest.json` 的 `assetBaseUrl`；本仓库不复制大模型权重。每次下载检查长度和 SHA-256，浏览器计算不上传输入。

这意味着在线推理依赖原站继续提供对应资源和 CORS 响应。若以后迁移模型托管，应更新基址并实测完整案例；文件内容变化时必须同步更新模型清单与核验记录。其它页面、数学渲染和冻结样本分析使用站内资源，不依赖原站在线。

Android 页面保留历史应用展示；APK 需要原仓库访问权限，因此提供原项目应用页入口，不将私有下载链接伪装为当前公共发布。

## 维护与核验

论文配图来自本地 `paper/figures/`。更新图件时，使用 Poppler 的 `pdftoppm` 导出网页预览并同步原始 PDF：

```bash
python3 scripts/export_site_figures.py --source /path/to/paper/figures
```

脚本只导出论文图 1–12，保持完整图面与原始 PDF；不上传整篇稿件。公共仓库不需要保留本地论文目录，日常网站构建直接使用已提交的图件，因此不依赖 Poppler。修改图面内容或尺寸后，也应同步对应页面的图注与图片宽高。

修改后运行构建、`check_pages.py` 以及 JS 语法检查。静态检查覆盖站内链接/锚点、公式字体、图件哈希、混淆矩阵、160 组子集预测、强度与类别 Shapley、分解一致性、推理运行时及案例输入协议。

本次迁移的浏览器截图与验证记录位于工作区 `validation/website/`，不随 Pages 分发。网站迁移来源见 [UPSTREAM.md](UPSTREAM.md)。
