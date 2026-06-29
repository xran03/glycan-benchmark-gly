# glycan-benchmark-gly

Glycan **immunogenicity** benchmark pipeline.  
Reference: Xu et al. 2024, [GlycanML](https://arxiv.org/pdf/2405.16206) (arXiv:2405.16206).

## Task

Binary classification: given a glycan (IUPAC-condensed), predict immunogenicity (0/1).  
Dataset: Train 1,026 | Valid 149 | Test 145 — downloaded from the GlycanML S3 bucket.

## ⚠️ Important Note on Results

Our results **cannot be directly compared** to the GlycanML paper baselines. Key differences:

| Aspect | GlycanML paper | This benchmark |
|--------|---------------|----------------|
| Graph format | Edge features for linkages | glycowork v1.9 (linkages as nodes) |
| Loss | Plain BCE, no class weighting | BCE + `pos_weight = neg/pos` |
| Training | Fixed 50 epochs, batch=256 | Early stopping, batch=64 |
| Vocab | Pre-built 211-token glycoword | Dataset-derived 181 tokens |

We also reproduced the GlycanML paper models using their **original torchdrug-based code**
(CPU-only, as torch 1.12 does not support H200/sm_90 CUDA). Results in `results/glycanml_original/`.

### Three-way comparison (AUROC on test set)

| Model  | Paper (reported) | GlycanML orig code (reproduced, CPU) | Our reimplementation |
|--------|-----------------|--------------------------------------|----------------------|
| GCN    | 0.749 | 0.862 | 0.988 |
| GAT    | 0.762 | 0.931 | 0.945 |
| GIN    | 0.778 | 0.958 | 0.981 |
| MPNN   | 0.785 | 0.968 | 0.984 |
| CNN    | 0.741 | 0.967 | 0.976 |
| ResNet | 0.756 | 0.358* | 0.954 |
| LSTM   | 0.728 | 0.982 | 0.987 |

> *ResNet in GlycanML original code failed to train properly on this dataset/setup.  
> See `results/glycanml_comparison.csv` for full details.

**Why are all results higher than the paper?** The dataset URL is the same but the S3 bucket
content appears to have changed since publication — the split sizes differ (Train 1026/1046,
Valid 149/131, Test 145/143). The current dataset likely produces easier splits than what the
paper evaluated on.

## Models

## Quick start

```bash
# 1. Create / activate environment
conda activate py3

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full benchmark (downloads dataset automatically)
python scripts/run_benchmark.py

# 4. Results are written to results/
#    sweetnet_metrics.json   — AUROC, AUPRC, F1, MCC
#    sweetnet_ranking.csv    — ranked vs GlycanML baselines
#    sweetnet_summary.txt    — human-readable report
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `results/` | Where to write outputs |
| `--max-epochs` | `50` | Training epochs (early-stopped) |
| `--lr` | `1e-3` | Learning rate |
| `--seed` | `42` | Random seed |
| `--csv-path` | *(auto-download)* | Local dataset CSV |
| `--config` | `configs/immunogenicity.yaml` | YAML config |

## GlycanML baselines

| Model    | AUROC | AUPRC |
|----------|-------|-------|
| MPNN     | 0.785 | 0.684 |
| GIN      | 0.778 | 0.677 |
| BERT     | 0.770 | 0.671 |
| GAT      | 0.762 | 0.660 |
| GCN      | 0.749 | 0.645 |
| CNN      | 0.741 | 0.639 |
| LSTM     | 0.728 | 0.621 |

## Repository layout

```
benchmark_gly/
├── benchmark/
│   ├── data/immunogenicity.py      # dataset download + splits
│   ├── models/sweetnet_classifier.py  # SweetNet encoder + MLP head
│   ├── evaluate.py                 # AUROC / AUPRC + baseline ranking
│   └── pipeline.py                 # end-to-end orchestrator
├── configs/immunogenicity.yaml     # hyperparameters
├── scripts/run_benchmark.py        # CLI entry point
├── results/                        # output directory (gitignored for models)
├── wiki/                           # KB wiki (AGENTS.md schema)
│   ├── index.md
│   ├── log.md
│   ├── map.md
│   ├── sources/
│   ├── entities/
│   └── concepts/
└── reference_kb/                   # frozen KB ref (XR/ml-vrd, not tracked)
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
