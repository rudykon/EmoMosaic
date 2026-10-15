[English (default)](README.md) | **简体中文**

<div align="center">
  <a href="https://github.com/rudykon/EmoMosaic">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/readme/emomosaic-horizontal-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="assets/readme/emomosaic-horizontal.svg">
      <img src="assets/readme/emomosaic-horizontal.svg" alt="EmoMosaic" width="460">
    </picture>
  </a>
  <p><strong>局部模态缺失下的多模态情感预测</strong></p>
  <p>文本 · 语音 · 视觉<br>RM-Net · BERT LoRA 文本专家 · 文本可用性门控融合</p>
  <p>
    <a href="https://huggingface.co/config-h/EmoMosaic"><img src="https://img.shields.io/badge/Hugging_Face-Models-7B68D8?style=flat-square&amp;logo=huggingface&amp;labelColor=263445" alt="Hugging Face 模型权重"></a>
    <a href="https://huggingface.co/datasets/config-h/EmoMosaic"><img src="https://img.shields.io/badge/Hugging_Face-Dataset-249D9C?style=flat-square&amp;logo=huggingface&amp;labelColor=263445" alt="Hugging Face 数据集"></a>
    <a href="requirements.txt"><img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white&amp;labelColor=263445" alt="Python 3.11"></a>
    <a href="requirements.txt"><img src="https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white&amp;labelColor=263445" alt="PyTorch 2.6.0"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/Code-MIT-EA8A77?style=flat-square&amp;labelColor=263445" alt="原创代码采用 MIT 许可证"></a>
  </p>
  <p>
    <a href="#overview">项目概览</a> ·
    <a href="#method">模型方法</a> ·
    <a href="#results">实验结果</a> ·
    <a href="#quick-start">快速开始</a> ·
    <a href="#reproduction">实验复现</a>
  </p>
</div>

**EmoMosaic** 将五种子 **RM-Net** 集成与五个 **BERT LoRA** 文本专家结合，利用文本、语音和视觉特征预测情感类别与连续强度。融合权重随可用文本比例调整，在文本证据缺失时降低文本专家的影响。

本仓库保存源码、配置、测试、项目网页及 README 使用的精选插图。**模型权重和数据托管于 Hugging Face**；论文正文、实验输出及生成缓存另行维护。默认 `README.md` 为英文，可通过顶部链接切换语言。

**项目网页：** [构建、预览与 GitHub Pages 部署说明](website/README.md)。沿用原 EmoMosaic 网页的布局，支持双语导航、实验结果和浏览器推理。

<a id="overview"></a>
## 🧩 项目概览

| 组成 | 主要功能 | 阅读入口 |
| --- | --- | --- |
| **多模态预测** | 融合文本、语音与视觉，显式处理缺失信息 | [模型与推理说明](code/src/p2p3/README.md) |
| **文本专家与融合** | 使用 BERT LoRA 补充预测，并按文本可用性调节权重 | [冻结预测器](code/src/p2p3/improved_predict.py) |
| **双目标解释** | 分析类别概率与情感强度的模态贡献及局部遮挡效应 | [解释引擎](code/src/p2p3/p3_fusion.py) |
| **特征与对齐** | 提取音视频特征，构建词级时间对齐 | [独立对齐流程](code/src/problem1/README.md) |
| **可复现资源** | 下载固定版本的模型与数据，并校验文件完整性 | [模型清单](scripts/model_release.json) · [数据下载器](scripts/download_data.py) |

<a id="method"></a>
## 🧠 模型方法

RM-Net 处理可用的三模态特征，BERT LoRA 专家提供补充文本预测。两个分支通过**分别设置的分类与回归权重**融合，分类端进一步应用固定的类别偏置校正。

<p align="center">
  <a href="assets/readme/method.png">
    <img src="assets/readme/method.png" alt="总体框架：三模态特征进入五种子 RM-Net，词元序列进入五个 BERT LoRA 专家，按文本可用度融合类别与强度预测。" width="900">
  </a>
  <br>
  <sub><strong>模型总体框架</strong> · 原论文图 5.1 · 点击查看原图</sub>
</p>

设可用文本比例为 `f_T`，冻结的文本专家权重分别为：分类 `0.80 × f_T²`，回归 `0.60 × f_T²`。文本完全不可用时，两个专家权重均为零。模型输出**负向／中性／正向**类别及 **[−3, 3]** 范围内的连续情感强度。

插图保留论文原始中文标签，固定版本的来源记录见[素材索引](assets/readme/README.md)。

<a id="results"></a>
## 📊 实验结果

下表来自 **RM-Net + BERT LoRA** 在赛题提供的 CMU-MOSEI 数据上的历史冻结评估。“混合缺失”对应固定的 40% 混合缺失评估视图。

| 划分 | 输入条件 | Accuracy ↑ | Macro-F1 ↑ | MAE ↓ | Pearson r ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| 验证集 | 完整输入 | 0.6442 | 0.6339 | 0.5652 | 0.6835 |
| 验证集 | 40% 混合缺失 | 0.6497 | 0.6394 | 0.5722 | 0.6638 |
| 测试集 | 完整输入 | **0.6809** | **0.6550** | 0.6099 | 0.6942 |
| 测试集 | 40% 混合缺失 | **0.6630** | **0.6341** | 0.6210 | 0.6861 |

测试集两视图平均 Macro-F1 为 **0.6446**，原 RM 辅助头为 **0.6208**，提升 **2.38 个百分点**。数值由[原始评估表](https://github.com/rudykon/EmoMosaic_GMCM2026/blob/50202237f5f38f1e44ee916c596d9ab57a92d94e/outputs/problem2/improvement/final_evaluation/headline_metrics.csv)四舍五入得到。

> **评估口径：** 历史测试集在开发阶段曾被查看，不能视为独立盲测。此处展示已有结果，并非本次重新运行评估。附件 3 没有真实标签，不能据其预测计算准确率。

<p align="center">
  <a href="assets/readme/missingness-gains.png">
    <img src="assets/readme/missingness-gains.png" alt="21 种缺失条件下，融合模型相对 RM 辅助头的 Macro-F1 百分点增益，以及相对 RM 核心的 MAE 降低量。" width="900">
  </a>
  <br>
  <sub><strong>缺失条件下的融合增益</strong> · 原论文图 5.7<br>727 条历史留出样本 · T / A / V 分别为文本／语音／视觉 · 正值表示改善</sub>
</p>

收益随缺失模式变化：严重文本缺失时分类效果可能下降，包括上图中 60% 的 T 和 TAV 条件。

<a id="explanations"></a>
## 🔎 解释示例

以附件 4 的 19 号样本为例，将词片段、语音轨道与视频关键帧放在同一候选时间轴上。局部遮挡衡量输入片段对预测类别概率的影响；解释流程同时支持连续强度归因。

<p align="center">
  <a href="assets/readme/explanation-sample19.png">
    <img src="assets/readme/explanation-sample19.png" alt="样本 19 的解释卡：候选词片段、文本／语音／视觉遮挡贡献、波形、基频、人脸检出和视频关键帧。" width="900">
  </a>
  <br>
  <sub><strong>共同时间轴上的三模态证据</strong> · 原论文图 6.13<br>橙色支持原预测类别，蓝色抵消；波形与原始帧仅辅助回看。</sub>
</p>

这些解释衡量输出层面的遮挡效应，不能视为因果效应，也不等价于在上下文编码前删除原始文本。

<a id="quick-start"></a>
## 🚀 快速开始

以下命令在类 Unix shell 中运行，需要 **Python 3.11** 和 **GNU Make**。默认安装 CPU 版 PyTorch。

### 1. 安装环境

```bash
git clone https://github.com/rudykon/EmoMosaic.git
cd EmoMosaic
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

### 2. 下载固定版本资源

```bash
make download-weights
make download-models
make download-data
```

| 命令 | 资源 | 说明 |
| --- | --- | --- |
| `make download-weights` | [EmoMosaic 模型发布](https://huggingface.co/config-h/EmoMosaic) → `code/models/` | 恢复固定版本的 11 个检查点／适配器文件及必要配置，校验 SHA-256，不覆盖已存在但校验不匹配的文件 |
| `make download-models` | BERT、CTC 与人脸模型 | 从上游下载固定版本并校验；最终情感预测使用 BERT，CTC 与人脸模型用于独立对齐流程 |
| `make download-data` | [既有数据发布](https://huggingface.co/datasets/config-h/EmoMosaic) → `data/source_data/` | 下载约 **3.80 GiB** 数据并逐项校验，公开下载无需令牌 |

<details>
<summary><strong>展开固定版本信息</strong></summary>

- **模型版本：** `15af38ee327811dcac3eca397746dba69266854c`；逐文件校验值见 [scripts/model_release.json](scripts/model_release.json)。
- **数据版本：** `c9c27b40a226f9f14ae05dde5b8ea4d5ae65ae87`；下载器恢复已有发布，不重复发布数据。

下载的权重、数据、缓存和实验输出保存在本地，由 [`.gitignore`](.gitignore) 排除。

</details>

### 3. 运行预测

```bash
make predict
```

使用最终冻结预测器推断无标签的附件 3 样本，结果保存到 **`validation/attachment3_predictions.csv`**。

<a id="reproduction"></a>
## 🔬 实验复现

### 重建缓存与评估

下载资源后，先在本地重建特征缓存，再运行评估：

```bash
.venv/bin/python code/src/p2p3/prepare.py --device cpu
make evaluate
```

[评估入口](code/src/p2p3/evaluate_deployed.py)在验证集和测试集上对照冻结融合模型与 RM 基线，覆盖完整输入、40% 混合缺失及 21 种缺失网格条件。指标、预测、协议记录及配对分类统计会写入 `code/outputs/reproduced/` 下的新目录。

<details>
<summary><strong>展开 RM-Net 新训练示例</strong></summary>

准备资源与缓存后，使用支持 CUDA 的 PyTorch 环境启动训练：

```bash
.venv/bin/python code/src/p2p3/train.py --variant A0 --seed 17 --device cuda:0 \
  --run-dir outputs/reproduced/rm_A0_seed17
```

该命令训练单次 RM-Net；复现完整部署融合还需要文本专家和冻结融合配置。详见[模型说明](code/src/p2p3/README.md)。

</details>

<details>
<summary><strong>展开词级对齐与数值解释流程</strong></summary>

独立对齐流程使用另一套依赖。完成资源下载后，建议安装到 `.venv-p1`：

```bash
python3.11 -m venv .venv-p1
.venv-p1/bin/python -m pip install -r requirements-alignment.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
make align-smoke
```

准备好资源、缓存与对齐环境后，运行数值解释：

```bash
make explain
```

该命令跳过绘图；需要重建词级对齐时，会调用独立对齐环境。若解释器不在 `.venv-p1/bin/python`，请设置 `COLING_ALIGNMENT_PYTHON`。

**独立对齐输出与主预测模型使用的官方特征不可互换。** 具体流程见[对齐说明](code/src/problem1/README.md)。

</details>

历史候选搜索、结果核验及部分绘图工具依赖额外实验文件；这些输出未随仓库提供。

<a id="repository"></a>
## 🗂️ 目录导航

```text
EmoMosaic/
├── README.md                    # 默认英文首页
├── README.zh-CN.md              # 中文说明
├── assets/readme/               # Logo、精选插图与来源记录
├── code/src/
│   ├── p2p3/                    # 模型、训练、推理与解释
│   ├── problem1/                # 特征提取与词级对齐
│   └── common/                  # 公共绘图与归档工具
├── scripts/
│   ├── download_weights.py      # 固定模型下载与 SHA-256 校验
│   ├── download_data.py         # 既有数据发布下载与校验
│   └── model_release.json       # 模型版本与逐文件校验值
├── Makefile                     # 下载、预测、评估与测试入口
├── requirements.txt             # 主环境依赖
├── requirements-alignment.txt   # 独立对齐环境依赖
└── LICENSE                      # 原创代码 MIT 许可证
```

`code/models/`、`data/source_data/`、`code/outputs/` 和 `validation/` 等下载及生成目录在本地创建。

<a id="checks-and-license"></a>
## ✅ 检查与许可

```bash
make test
make check-weights
```

原创代码采用 [MIT 许可证](LICENSE)。数据、第三方模型及依赖分别遵循其原有条款；数据使用条件见[数据仓库 TERMS.md](https://huggingface.co/datasets/config-h/EmoMosaic/blob/main/TERMS.md)。精选论文插图与品牌素材来自 [EmoMosaic 竞赛项目](https://github.com/rudykon/EmoMosaic_GMCM2026)，来源详见[素材索引](assets/readme/README.md)。

<p align="center">
  <a href="#overview">返回项目概览 ↑</a>
</p>
