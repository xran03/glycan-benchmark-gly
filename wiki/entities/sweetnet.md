# Entity: SweetNet

**Type**: Graph Neural Network encoder  
**Package**: [glycowork](https://github.com/BojarLab/glycowork) ≥ 1.9.0  
**Reference**: Bojar et al. 2021

## Description

SweetNet is a 3-layer GraphConv GNN trained on 1,075 glycan-species association labels. It maps an IUPAC-condensed glycan graph to a 128-dimensional embedding.

In this benchmark SweetNet is used as a **frozen feature extractor** — its weights are not updated during fine-tuning.

## Architecture

```
Embedding(lib_size+1, 128)
→ GraphConv × 3 (128 hidden, LeakyReLU)
→ GlobalMeanPool
→ FC 128→1024 → BN → LeakyReLU
→ FC 1024→128  → BN              ← 128-d output embedding
→ FC 128→1     (dropped, head replaces this)
```

## Usage in benchmark

```python
from glycowork.ml.models import prep_model
from glycowork.ml.inference import glycans_to_emb

encoder = prep_model("SweetNet", num_classes=1075, trained=True)
emb_df  = glycans_to_emb(glycans, encoder, rep=True)  # (N, 128)
```

## Cross-references

- [Concept: Immunogenicity Benchmark](../concepts/immunogenicity-benchmark.md)
- [Source: GlycanML](../sources/2026-06-24--glycanml-immunogenicity.md)
