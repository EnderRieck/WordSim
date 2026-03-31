# 多数据集微调说明

本文档说明如何用配置文件组合 `LCQMC`、`STS-B (zh)`、`PAWSX (zh)` 和英文 `MS MARCO` triplets，对多语言 CrossEncoder 基座 `cross-encoder/ms-marco-MiniLM-L-6-v2` 做混合微调，并在当前服务中与 `BAAI/bge-reranker-large` 做对比。

## 1. 环境准备

先安装后端依赖：

```bash
cd backend
pip install -r requirements.txt
```

依赖安装完成后，回到仓库根目录：

```bash
cd /mnt/DataFlow/lz/proj/agentgroup/ziyi/BUPT/IIN/BERT
```

## 2. 准备混合数据

运行：

```bash
python training/prepare_datasets.py --config training/configs/multitask_default.json
```

默认配置文件位置：

```text
training/configs/multitask_default.json
```

你可以直接在这个 JSON 里改参数，核心命名统一由 `model_name` 控制：

- 模型目录：`artifacts/models/{model_name}`
- 报告文件名：`{model_name}_comparison.json/.md`
- 当前激活微调模型指针：`artifacts/models/active_model_name.txt`
- 数据导出目录：`data_output_dir`

当前支持：

- `model_name`
- `training.learning_rate`
- `training.epochs`
- `training.train_batch_size`
- `training.eval_batch_size`
- `training.max_length`
- `training.warmup_ratio`
- `training.base_model`
- `evaluation.report_dir`

默认会混合这些数据集：

- `LCQMC`
- `STS-B (zh)`
- `PAWSX (zh)`
- `MS MARCO` 英文 triplets

并导出到：

```text
artifacts/data/multitask/train.tsv
artifacts/data/multitask/validation.tsv
artifacts/data/multitask/test.tsv
artifacts/data/multitask/metadata.json
```

导出的 TSV 字段为：

```text
dataset_name    text1    text2    label
```

如果你只想用某几个数据集，直接改配置文件里对应数据集的 `enabled` 字段即可。

例如只保留：

- `lcqmc`
- `stsb_zh`
- `msmarco_en`

就把 `pawsx_zh` 改成：

```json
"enabled": false
```

仓库里也提供了一个只用 LCQMC 的配置：

```text
training/configs/lcqmc_only.json
```

对于像 `MS MARCO triplets` 这类只有 `train` 源 split 的数据，脚本会按照配置里的 `train_examples / validation_examples / test_examples` 自动从训练集切出验证集和测试集。

## 3. 开始微调

使用混合数据目录训练：

```bash
python training/train_cross_encoder.py --config training/configs/multitask_default.json
```

默认配置：

- 基座模型：`cross-encoder/ms-marco-MiniLM-L-6-v2`
- epoch：`3`
- `train_batch_size=32`
- `eval_batch_size=64`
- `max_length=256`
- 学习率：`5e-6`

如果 `validation.tsv` 里包含连续标签，例如 `STS-B`，训练脚本会自动切换到相关性 evaluator；如果全是 `0/1` 标签，则继续使用二分类 evaluator。

训练输出目录：

```text
artifacts/models/{model_name}
```

训练完成后，这个目录里应该至少能看到：

```text
config.json
tokenizer.json
model.safetensors 或 pytorch_model.bin
training_config.json
experiment_config.json
```

如果这里只有 `eval/` 和 `training_config.json`，说明模型没有正确导出，需要重新运行训练。

如果你想临时覆盖配置文件里的值，也可以直接传参，例如：

```bash
python training/train_cross_encoder.py \
  --config training/configs/multitask_default.json \
  --epochs 5 \
  --train-batch-size 16 \
  --learning-rate 1e-5
```

## 4. 评估三模型对比

训练完成后运行：

```bash
python training/eval_models.py --config training/configs/multitask_default.json
```

会对以下三个模型统一评估：

- `bge_zh` -> `BAAI/bge-reranker-large`
- `minilm_zero_shot` -> `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `minilm_finetuned` -> 微调后的本地模型

评估结果输出到 `evaluation.report_dir` 指定目录，文件名会自动由 `model_name` 派生。

默认配置会生成：

```text
artifacts/reports/{model_name}_comparison.json
artifacts/reports/{model_name}_comparison.md
```

混合测试集会输出两类指标：

- 全量相关性指标：`Pearson`、`Spearman`
- 二分类子集指标：`Accuracy`、`F1`、`AUC`

并且会按数据集分别拆开显示。

## 5. 在服务中使用

启动服务：

```bash
cd backend
uvicorn main:app --reload
```

然后访问：

```text
http://localhost:8000
```

服务支持三种模型选择：

- `bge_zh`
- `minilm_zero_shot`
- `minilm_finetuned`

如果 `minilm_finetuned` 还没训练完成，前端和接口会显示“模型未就绪”。

## 6. 常见问题

### 1. 为什么看不到微调模型？

因为服务会检查这个目录是否存在：

```text
artifacts/models/{model_name}
```

只有训练成功并导出到该目录后，`minilm_finetuned` 才会变成可用状态。

训练脚本还会同时更新：

```text
artifacts/models/active_model_name.txt
```

后端会优先读取这个文件，把对应模型作为当前可用的微调模型。

### 2. 为什么训练跑不起来？

优先检查这几项：

- 是否已经执行 `pip install -r backend/requirements.txt`
- 是否能正常下载 Hugging Face 模型和数据集
- GPU 显存是否足够

### 3. 训练前后怎么对比？

直接看配置里指定名字的 `.md` 报告文件，它会汇总：

- 多语言基座未微调结果
- 混合数据微调后结果
- 中文强基线 `bge-reranker-large` 结果
