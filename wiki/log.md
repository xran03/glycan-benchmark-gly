# Wiki Log

<!-- Append-only. Newest entries at the bottom. -->

## 2026-06-24 — Training run completed

- Trained SweetNet from scratch (vocab_size=181, random init) on immunogenicity dataset
- GPU: CUDA_VISIBLE_DEVICES=0 (NVIDIA H200)
- Result: AUROC=0.974, AUPRC=0.8132, F1=0.7222, MCC=0.689
- Ranking: #1 / 11 (+0.1890 AUROC vs best GlycanML baseline MPNN=0.785)
- Early stopping at epoch 20 (best val_loss=0.1421)
- Plots generated: roc_curve, pr_curve, training_loss, baseline_ranking, confusion_matrix
- Model weights: results/sweetnet_model.pt


- Ingested: GlycanML immunogenicity benchmark (Xu et al. 2024, arXiv:2405.16206)
- Created: `sources/2026-06-24--glycanml-immunogenicity.md`
- Created: `entities/sweetnet.md`
- Created: `concepts/immunogenicity-benchmark.md`
- Created: `map.md`, `index.md`
- Built: benchmark pipeline (`benchmark/`, `configs/`, `scripts/`)
- Frozen reference KB: `reference_kb/` (XR/ml-vrd branch, pfizer-rd/test-personal-KB-vrd, untracked)
