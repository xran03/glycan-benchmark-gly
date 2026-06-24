# GlycanML Immunogenicity Benchmark

**Ingested**: 2026-06-24  
**Source**: Xu et al. 2024, *GlycanML: A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning*  
**arXiv**: https://arxiv.org/pdf/2405.16206  
**GitHub**: https://github.com/GlycanML/GlycanML

---

## Dataset

| Split | Samples | Positive (%) |
|-------|---------|--------------|
| Train | 1,046   | ~38 %        |
| Valid | 131     | ~38 %        |
| Test  | 143     | ~38 %        |

- **Task**: Binary classification — immunogenic (1) vs. non-immunogenic (0)
- **Input**: IUPAC-condensed glycan strings
- **Label**: `immunogenicity` column
- **Download**: `https://torchglycan.s3.us-east-2.amazonaws.com/downstream/glycan_immunogenicity.csv`
- **MD5**: `5ee0814b23304f67247e85786a8b4688`

---

## Baseline Results (from paper)

Primary metrics: **AUROC** and **AUPRC** (higher is better).

| Model    | Type     | AUROC | AUPRC |
|----------|----------|-------|-------|
| MPNN     | Graph    | 0.785 | 0.684 |
| GIN      | Graph    | 0.778 | 0.677 |
| BERT     | Sequence | 0.770 | 0.671 |
| GAT      | Graph    | 0.762 | 0.660 |
| ResNet   | Sequence | 0.756 | 0.654 |
| RGCN     | Graph    | 0.758 | 0.655 |
| GCN      | Graph    | 0.749 | 0.645 |
| CompGCN  | Graph    | 0.752 | 0.649 |
| CNN      | Sequence | 0.741 | 0.639 |
| LSTM     | Sequence | 0.728 | 0.621 |

**Competitive target**: AUROC ≥ 0.785, AUPRC ≥ 0.684 (match MPNN).

---

## Key Findings (paper)

- Graph-based models (GIN, MPNN) slightly outperform sequence-based (CNN, LSTM)
- Multi-task learning consistently improves performance across tasks
- Best single-task graph model: MPNN

---

## Cross-references

- [Entity: SweetNet](../entities/sweetnet.md) — encoder used in this benchmark
- [Concept: Immunogenicity Benchmark](../concepts/immunogenicity-benchmark.md)
