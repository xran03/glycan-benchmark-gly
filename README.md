# glycan-benchmark-gly

Glycan **immunogenicity** benchmark pipeline, evaluated against the
[GlycanML](https://github.com/GlycanML/GlycanML) public leaderboard
(Xu et al. 2024, [arXiv:2405.16206](https://arxiv.org/pdf/2405.16206)).

## Task

Binary classification: given a glycan (IUPAC-condensed), predict immunogenicity (0/1).  
Dataset: Train 1,046 | Valid 131 | Test 143 — downloaded automatically from the GlycanML S3 bucket.

## Model

**SweetNet** (glycowork ≥ 1.9.0) as a frozen graph-encoder → fine-tuned MLP head.

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
