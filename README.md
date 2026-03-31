# BERT CrossEncoder 句子相似度计算

基于 CrossEncoder 的句子相似度系统，支持在线推理、注意力可视化、多模型对比，以及配置式多数据集微调。

## 功能特点

- 默认服务模型：`BAAI/bge-reranker-large`
- 对比模型：
  - `BAAI/bge-reranker-large`
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - 混合数据微调后的 `MiniLM`
- 支持配置式选择训练数据集：
  - `LCQMC`
  - `STS-B (zh)`
  - `PAWSX (zh)`
  - 英文 `MS MARCO` triplets

## 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

## 启动服务

```bash
cd backend
uvicorn main:app --reload
```

访问 `http://localhost:8000`。

## 准备多数据集训练数据

```bash
cd /mnt/DataFlow/lz/proj/agentgroup/ziyi/BUPT/IIN/BERT
python training/prepare_datasets.py --config training/configs/multitask_default.json
```

默认输出目录：

```text
artifacts/data/multitask/
```

## 开始训练

```bash
python training/train_cross_encoder.py --dataset-dir artifacts/data/multitask
```

## 评估

```bash
python training/eval_models.py --dataset-dir artifacts/data/multitask
```

评估报告默认输出到：

```text
artifacts/reports/multitask_comparison.json
artifacts/reports/multitask_comparison.md
```

## 文档

- 训练和配置说明见 `TRAINING.md`
