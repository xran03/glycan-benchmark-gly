# Concept: Glycan Immunogenicity Benchmark

## Definition

Binary classification task: given a glycan structure (IUPAC-condensed), predict whether it will elicit an immune response (label = 1) or not (label = 0).

## Evaluation Protocol

| Aspect | Detail |
|--------|--------|
| Splits | Fixed train/valid/test (1046/131/143) from GlycanML |
| Primary metrics | AUROC, AUPRC |
| Secondary metrics | Accuracy, F1, MCC (threshold = 0.5) |
| Positive-class weighting | BCEWithLogitsLoss with `pos_weight = neg_count/pos_count` |

## Why AUROC + AUPRC?

- **AUROC** measures overall discriminative ability, threshold-independent
- **AUPRC** is preferred for imbalanced datasets (~38 % positive rate)
- Both reported in GlycanML paper, enabling direct comparison

## Our Approach

1. Encode glycan strings → 128-d embeddings via frozen SweetNet
2. Fine-tune a 2-layer MLP head (128→64→1) with early stopping
3. Evaluate on the held-out test split

## Cross-references

- [Entity: SweetNet](../entities/sweetnet.md)
- [Source: GlycanML baselines](../sources/2026-06-24--glycanml-immunogenicity.md)
