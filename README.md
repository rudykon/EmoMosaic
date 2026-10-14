# EmoMosaic

局部模态缺失下的多模态情感预测：RM-Net、BERT LoRA 文本专家与文本可用性门控融合。

- **代码**：[GitHub](https://github.com/rudykon/EmoMosaic)
- **模型权重**：[Hugging Face 模型仓库](https://huggingface.co/config-h/EmoMosaic)
- **现有数据集**：[Hugging Face 数据仓库](https://huggingface.co/datasets/config-h/EmoMosaic)

GitHub 仅提供源码、配置、测试和使用说明。数据、模型权重、实验输出、论文及其构建/导出材料、`docs/` 和补充材料不在 Git 中；下载的资源保存在本地并由 `.gitignore` 排除。

## 安装

以下命令从仓库根目录运行，使用 Python 3.11：

```bash
git clone https://github.com/rudykon/EmoMosaic.git
cd EmoMosaic
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

## 下载资源与预测

```bash
make download-weights
make download-models
make download-data
make predict
```

`download-weights` 将 HF 上固定版本的 11 个检查点/适配器及必要配置恢复到 `code/models/`，逐项校验 SHA-256。已存在但不匹配的文件不会被覆盖。版本与校验值在 [scripts/model_release.json](scripts/model_release.json) 中。

`download-models` 从原始来源获取固定版本的 BERT、CTC 与人脸模型并校验；最终预测只使用 BERT，另外两项服务于独立的词级对齐流程。

`download-data` 使用已经发布的数据版本 `c9c27b40a226f9f14ae05dde5b8ea4d5ae65ae87`，将约 3.80 GiB 数据恢复到 `data/source_data/` 并逐项校验，不重复发布数据。公开下载无需令牌。

`predict` 对无标签的附件3样本推断，输出 `validation/attachment3_predictions.csv`。这些样本没有真实标签，不能由此计算准确率。

## 源码入口

| 位置 | 内容 |
| --- | --- |
| `code/src/p2p3/` | 模型、训练、推理、评估、缺失模拟与解释 |
| `code/src/problem1/` | 独立的音视频特征提取与词级对齐 |
| `code/src/common/` | 通用绘图与归档工具 |
| `scripts/download_data.py` | 下载并校验既有数据发布 |
| `scripts/download_weights.py` | 下载并校验 HF 模型发布 |

特征缓存需在下载资源后本地重建：

```bash
.venv/bin/python code/src/p2p3/prepare.py --device cpu
make evaluate
```

新训练示例：

```bash
.venv/bin/python code/src/p2p3/train.py --variant A0 --seed 17 --device cuda:0 \
  --run-dir outputs/reproduced/rm_A0_seed17
```

独立对齐流程使用另一套依赖，建议安装到 `.venv-p1`：

```bash
python3.11 -m venv .venv-p1
.venv-p1/bin/python -m pip install -r requirements-alignment.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
make align-smoke
```

历史候选搜索、结果核验和部分图表工具依赖另行生成的实验文件；仓库未附带这些输出。历史留出集曾在开发阶段被查看，不能视为独立盲测。词级对齐输出与主预测模型使用的官方特征不可互换。

## 检查与许可

```bash
make test
make check-weights
```

原创代码采用 [MIT 许可证](LICENSE)。现有数据、第三方模型及依赖分别遵循其原有条款；数据说明与使用条件见[数据仓库 TERMS.md](https://huggingface.co/datasets/config-h/EmoMosaic/blob/main/TERMS.md)。
