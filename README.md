# glycan-benchmark-gly

Glycan **immunogenicity** benchmark pipeline.  
Reference: Xu et al. 2024, [GlycanML](https://arxiv.org/pdf/2405.16206) (arXiv:2405.16206).

## Task

Binary classification: given a glycan (IUPAC-condensed), predict immunogenicity (0/1).  
Dataset: Train 1,026 | Valid 149 | Test 145 — downloaded from the GlycanML S3 bucket.

## Hardware & Environment

All models were trained on **NVIDIA H200 GPU (143 GB VRAM)**:

| Model group | Training setup |
|-------------|---------------|
| GCN, GAT, GIN, MPNN, CNN, ResNet, LSTM, SweetNet, GlycanAA | Single GPU — `CUDA_VISIBLE_DEVICES=0`, `py3` env (PyTorch 2.12.1+cu130) |
| GlycanGT-large | **2-GPU DDP** — `CUDA_VISIBLE_DEVICES=0,2`, `torchrun --nproc_per_node=2`, same `py3` env |
| GlycanML original code (reproduced) | **CPU only** — `tdrug` env (PyTorch 1.12+cu113, torchdrug 0.2.1 incompatible with H200/sm_90) |

## Results (GPU, no pos_weight)

| Rank | Model | AUROC | AUPRC | F1 | MCC |
|------|-------|-------|-------|-----|-----|
| 1 | LSTM | **0.9868** | 0.9258 | 0.8095 | 0.7786 |
| 2 | SweetNet | 0.9848 | 0.8646 | 0.7059 | 0.6818 |
| 3 | GIN | 0.9840 | 0.8971 | 0.8889 | 0.8764 |
| 4 | MPNN | 0.9816 | 0.8327 | 0.8095 | 0.7786 |
| 5 | ResNet | 0.9780 | 0.8650 | 0.6875 | 0.6783 |
| 6 | GCN | 0.9776 | 0.8464 | 0.7500 | 0.7100 |
| 7 | CNN | 0.9756 | 0.8665 | 0.6286 | 0.5865 |
| 8 | GAT | 0.9644 | 0.8253 | 0.7917 | 0.7670 |
| 9 | GlycanAA | 0.9488 | 0.6953 | 0.5294 | 0.4787 |
| 10 | GlycanGT-large (supervised) | 0.9344 | 0.6633 | 0.5854 | 0.5173 |

> GlycanAA: heterogeneous all-atom GNN, torchdrug-free reimplementation (PyG RGCNConv).  
> GlycanGT: TokenGT architecture (88M params), trained **from scratch** without pretrained weights
> (HuggingFace `Akikitani295/GlycanGT` blocked by corporate firewall).

## ⚠️ Why our results are higher than the paper

**These results cannot be directly compared to the GlycanML paper baselines.** There are two root causes:

### 1. Dataset changed on S3 (main cause)

| | Paper (2024) | Current S3 |
|--|-------------|-----------|
| Train | 1,046 | 1,026 |
| Valid | 131 | 149 |
| Test | 143 | 145 |
| Train positive rate | unknown | **58.8%** |
| Test positive rate | unknown | **13.8%** |

The current S3 split places 60% of positives in training but only 14% in the test set (20 positives out of 145). With so few positive–negative pairs in the test set (20×125 = 2,500), AUROC is much easier to inflate than with a balanced test set. **Even the original GlycanML torchdrug code reproduces this inflation on the current data** (GCN: 0.862 vs paper 0.749).

### 2. Implementation differences

| Aspect | GlycanML paper | This benchmark |
|--------|---------------|----------------|
| Graph format | Edge features for linkages | glycowork v1.9 (linkages as nodes) |
| Loss | Plain BCE, no weighting | Plain BCE, no weighting ✓ (fixed) |
| Vocab | Pre-built 143-unit monosaccharide | Dataset-derived 181 tokens |
| Training | Fixed 50 epochs, batch=256 | Early stopping (patience=10), variable batch |

### Three-way comparison (AUROC on test set)

| Model | Paper (reported) | GlycanML orig code (CPU, reproduced) | Our GPU impl |
|-------|-----------------|--------------------------------------|-------------|
| GCN | 0.749 | 0.862 | 0.9776 |
| GAT | 0.762 | 0.931 | 0.9644 |
| GIN | 0.778 | 0.958 | 0.9840 |
| MPNN | 0.785 | 0.968 | 0.9816 |
| CNN | 0.741 | 0.967 | 0.9756 |
| ResNet | 0.756 | 0.358* | 0.9780 |
| LSTM | 0.728 | 0.982 | 0.9868 |

> *ResNet in GlycanML original code failed to train properly (AUROC=0.358).  
> See `results/glycanml_comparison.csv` for details.

## Models

| Model | Type | Framework | Notes |
|-------|------|-----------|-------|
| SweetNet | Graph GNN | glycowork | Official glycan GNN |
| GCN / GAT / GIN / MPNN | Graph GNN | PyG | Standard graph networks |
| CNN / ResNet / LSTM | Sequence | PyTorch | Token-based sequence models |
| GlycanAA | Heterogeneous GNN | PyG (torchdrug-free) | All-atom, 3-stream RGCN |
| GlycanGT-large | Graph Transformer | PyTorch DDP | TokenGT 88M params, no pretraining |

## Quick Start

```bash
# Check GPU availability first
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# 8 baseline models (single GPU)
CUDA_VISIBLE_DEVICES=0 python scripts/run_all_models.py --seed 42

# GlycanAA (single GPU)
CUDA_VISIBLE_DEVICES=0 python scripts/run_glycanaa.py --output-dir results/GlycanAA --epochs 50

# GlycanGT large (2-GPU DDP)
CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 --master_port=29500 \
    scripts/run_glycangt.py --model-size large --epochs 100 --batch-size 32 --lr 5e-5
```

## Environments

| Env | Python | PyTorch | CUDA | Use |
|-----|--------|---------|------|-----|
| `py3` | 3.11 | 2.12.1+cu130 | 13.0 | **All GPU training** |
| `tdrug` | 3.9 | 1.12.0+cu113 | — | GlycanML orig code (CPU only on H200) |

## Repository layout

```
benchmark_gly/
├── benchmark/
│   ├── data/immunogenicity.py          # dataset download + vocab (181 tokens)
│   ├── models/
│   │   ├── base_trainer.py             # shared trainer (graph/seq, early stop)
│   │   ├── gnn_models.py               # GCN, GAT, GIN, MPNN
│   │   ├── seq_models.py               # CNN, ResNet, LSTM
│   │   ├── sweetnet_classifier.py      # SweetNet
│   │   ├── glycanaa_graph.py           # GlycanAA heterogeneous graph builder
│   │   ├── glycanaa_model.py           # GlycanAA 3-stream RGCN (torchdrug-free)
│   │   └── glycanaa_trainer.py         # GlycanAA training wrapper
│   └── evaluate.py                     # metrics + baseline comparison
├── scripts/
│   ├── run_all_models.py               # all 8 baseline models
│   ├── run_glycanaa.py                 # GlycanAA benchmark
│   ├── run_glycangt.py                 # GlycanGT DDP benchmark
│   └── run_glycanml_original.py        # GlycanML torchdrug repro (CPU)
├── GlycanAA/                           # git submodule (kasawa1234/GlycanAA)
├── GlycanGT/                           # git submodule (matsui-lab/GlycanGT)
├── GlycanML_ref/                       # GlycanML reference code
└── results/
    ├── {Model}/                        # per-model metrics, arrays, history
    ├── plot/                           # ROC/PR/loss plots + combined bar chart
    ├── all_models_ranking.csv
    ├── all_models_summary.txt
    └── glycanml_comparison.csv         # 3-way comparison table
```

## Reference

```bibtex
@article{xu2024glycanml,
  title   = {GlycanML: A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning},
  author  = {Xu, Minghao and others},
  journal = {arXiv:2405.16206},
  year    = {2024}
}
```

