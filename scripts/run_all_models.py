#!/usr/bin/env python3
"""
Run all glycan immunogenicity benchmark models sequentially on GPU 0.

Models benchmarked:
  Graph-based : SweetNet, GCN, GAT, GIN, MPNN
  Seq-based   : CNN, ResNet, LSTM

Outputs (per model in results/<model>/):
  <model>_metrics.json   AUROC, AUPRC, F1, MCC
  <model>_ranking.csv    model vs all GlycanML baselines
  <model>_summary.txt    human-readable report
  <model>_history.json   epoch-level losses
  <model>_arrays.npz     y_true, y_prob for downstream use

Combined outputs in results/:
  all_models_ranking.csv    one row per model
  all_models_summary.txt    full comparison table
  plot/all_models_auroc.png bar chart across all models

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_all_models.py
    CUDA_VISIBLE_DEVICES=0 python scripts/run_all_models.py --models GCN,GAT,GIN
    CUDA_VISIBLE_DEVICES=0 python scripts/run_all_models.py --skip-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from benchmark.data.immunogenicity import ImmunogenicityDataset
from benchmark.evaluate import (
    GLYCANML_BASELINES,
    compute_metrics,
    rank_against_baselines,
    save_results,
)
from benchmark.models.gnn_models import GNN_REGISTRY, DEVICE
from benchmark.models.seq_models import SEQ_REGISTRY
from benchmark.models.base_trainer import Trainer
from benchmark.models.sweetnet_classifier import SweetNetClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── hyperparameters per model (tuned to GlycanML configs) ───────────────────
MODEL_HPARAMS: dict[str, dict] = {
    "SweetNet": dict(lr=5e-4, weight_decay=1e-4, max_epochs=50, batch_size=64,  patience=10, hidden_dim=128),
    "GCN":      dict(lr=5e-4, weight_decay=1e-3, max_epochs=50, batch_size=256, patience=10),
    "GAT":      dict(lr=5e-4, weight_decay=1e-3, max_epochs=50, batch_size=256, patience=10),
    "GIN":      dict(lr=5e-4, weight_decay=1e-3, max_epochs=50, batch_size=256, patience=10),
    "MPNN":     dict(lr=5e-4, weight_decay=1e-3, max_epochs=50, batch_size=128, patience=10),
    "CNN":      dict(lr=2e-4, weight_decay=1e-4, max_epochs=50, batch_size=256, patience=10),
    "ResNet":   dict(lr=2e-4, weight_decay=1e-4, max_epochs=50, batch_size=256, patience=10),
    "LSTM":     dict(lr=5e-5, weight_decay=1e-4, max_epochs=50, batch_size=64,  patience=10),
}

ALL_MODELS = list(MODEL_HPARAMS.keys())


def run_one(
    model_name: str,
    train_df, valid_df, test_df,
    vocab: dict[str, int],
    output_dir: Path,
    seed: int,
) -> dict[str, float]:
    """Train one model, evaluate, save outputs; return metrics dict."""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    hp = {**MODEL_HPARAMS[model_name], "seed": seed}

    logger.info("=" * 60)
    logger.info("MODEL: %s", model_name)
    logger.info("=" * 60)

    # ── SweetNet uses its own Classifier ──────────────────────────────────
    if model_name == "SweetNet":
        clf = SweetNetClassifier(
            vocab=vocab,
            lr=hp["lr"], weight_decay=hp["weight_decay"],
            max_epochs=hp["max_epochs"], batch_size=hp["batch_size"],
            patience=hp["patience"], seed=hp["seed"],
            hidden_dim=hp.get("hidden_dim", 128),
        )
        history = clf.fit(train_df, valid_df)
        y_prob  = clf.predict_proba(test_df["glycan"].tolist())
        y_true  = test_df["label"].values
        metrics = compute_metrics(y_true, y_prob)
        clf.save(model_dir / "model.pt")

    # ── GNN models ────────────────────────────────────────────────────────
    elif model_name in GNN_REGISTRY:
        ModelCls = GNN_REGISTRY[model_name]
        vocab_size = len(vocab)
        model = ModelCls(vocab_size=vocab_size).to(DEVICE)
        trainer = Trainer(
            model, mode="graph", vocab=vocab,
            lr=hp["lr"], weight_decay=hp["weight_decay"],
            max_epochs=hp["max_epochs"], batch_size=hp["batch_size"],
            patience=hp["patience"], seed=hp["seed"],
        )
        history = trainer.fit(train_df, valid_df)
        metrics, y_true, y_prob = trainer.evaluate(test_df)
        trainer.save(model_dir / "model.pt")

    # ── Sequence models ───────────────────────────────────────────────────
    elif model_name in SEQ_REGISTRY:
        from benchmark.models.seq_models import GlycanTokenizer
        ModelCls = SEQ_REGISTRY[model_name]
        tokenizer = GlycanTokenizer(vocab)
        model = ModelCls(vocab_size=tokenizer.vocab_size).to(DEVICE)
        trainer = Trainer(
            model, mode="seq", vocab=vocab,
            lr=hp["lr"], weight_decay=hp["weight_decay"],
            max_epochs=hp["max_epochs"], batch_size=hp["batch_size"],
            patience=hp["patience"], seed=hp["seed"],
        )
        history = trainer.fit(train_df, valid_df)
        metrics, y_true, y_prob = trainer.evaluate(test_df)
        trainer.save(model_dir / "model.pt")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # ── persist ─────────────────────────────────────────────────────────
    ranking_df = rank_against_baselines(metrics, model_label=model_name)
    save_results(metrics, ranking_df, model_dir, run_name=model_name)
    np.savez(model_dir / f"{model_name}_arrays.npz", y_true=y_true, y_prob=y_prob)
    (model_dir / f"{model_name}_history.json").write_text(json.dumps(history, indent=2))

    # ── plots ────────────────────────────────────────────────────────────
    plot_dir = output_dir / "plot"
    try:
        from results.plot.plot_results import plot_roc, plot_pr, plot_training_loss, plot_confusion_matrix
        plot_roc(y_true, y_prob, metrics["AUROC"], plot_dir / model_name)
        plot_pr(y_true, y_prob, metrics["AUPRC"], plot_dir / model_name)
        plot_training_loss(history, plot_dir / model_name)
        plot_confusion_matrix(y_true, y_prob, plot_dir / model_name)
    except Exception as e:
        logger.warning("Per-model plotting failed (non-fatal): %s", e)

    logger.info("%s → AUROC=%.4f  AUPRC=%.4f  F1=%.4f",
                model_name, metrics["AUROC"], metrics["AUPRC"], metrics["F1"])
    return metrics


def build_combined_summary(all_metrics: dict[str, dict], output_dir: Path) -> None:
    """Write combined ranking CSV + text summary + AUROC bar chart."""
    import pandas as pd

    rows = []
    for name, m in all_metrics.items():
        rows.append({"Model": name, "Source": "this benchmark",
                     "AUROC": m["AUROC"], "AUPRC": m["AUPRC"],
                     "F1": m.get("F1", float("nan")),
                     "MCC": m.get("MCC", float("nan"))})
    for bl_name, bl in GLYCANML_BASELINES.items():
        rows.append({"Model": bl_name, "Source": "GlycanML (paper)",
                     "AUROC": bl["AUROC"], "AUPRC": bl["AUPRC"],
                     "F1": float("nan"), "MCC": float("nan")})

    df = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"
    df.to_csv(output_dir / "all_models_ranking.csv")

    # text summary
    lines = [
        "=" * 70,
        "GLYCAN IMMUNOGENICITY BENCHMARK — ALL MODELS SUMMARY",
        "=" * 70,
        "",
        df.to_string(),
        "",
        "Reference: Xu et al. 2024, arXiv:2405.16206",
        f"Models trained: {', '.join(all_metrics.keys())}",
    ]
    (output_dir / "all_models_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    # AUROC bar chart
    try:
        _plot_combined_auroc(df, output_dir / "plot")
    except Exception as e:
        logger.warning("Combined AUROC plot failed: %s", e)


def _plot_combined_auroc(df, plot_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plot_dir.mkdir(parents=True, exist_ok=True)

    PALETTE = {"this benchmark": "#E87722", "GlycanML (paper)": "#868E96"}

    df_sorted = df.sort_values("AUROC")
    colors = [PALETTE.get(s, "#868E96") for s in df_sorted["Source"]]

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 7))
    for ax, col in zip(axes, ["AUROC", "AUPRC"]):
        bars = ax.barh(df_sorted["Model"], df_sorted[col],
                       color=colors, edgecolor="white", height=0.6)
        ax.set_xlabel(col)
        ax.set_title(col)
        ax.set_xlim(0.5, 1.0)
        for bar, val in zip(bars, df_sorted[col]):
            if not (val != val):  # skip NaN
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=8)

    axes[0].set_ylabel("Model")
    fig.suptitle(
        "Glycan Immunogenicity — Full Benchmark Ranking\nvs GlycanML Baselines (Xu et al. 2024)",
        fontsize=12,
    )
    legend_els = [
        Patch(color=PALETTE["this benchmark"],  label="This benchmark"),
        Patch(color=PALETTE["GlycanML (paper)"], label="GlycanML paper"),
    ]
    fig.legend(handles=legend_els, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_dir / "all_models_auroc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Combined AUROC plot → %s/all_models_auroc.png", plot_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run all glycan immunogenicity benchmarks")
    ap.add_argument("--output-dir",    default="results")
    ap.add_argument("--models",        default=None,
                    help=f"Comma-separated subset of {ALL_MODELS}. Default: all")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip models whose metrics.json already exists")
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    models = args.models.split(",") if args.models else ALL_MODELS

    logger.info("Loading dataset and building vocabulary …")
    ds = ImmunogenicityDataset()
    logger.info(ds.stats())
    train_df, valid_df, test_df = ds.split()
    vocab = ds.vocab

    all_metrics: dict[str, dict] = {}

    # Load pre-existing results so they appear in the combined summary
    for m in ALL_MODELS:
        existing = output_dir / m / f"{m}_metrics.json"
        if existing.exists():
            all_metrics[m] = json.loads(existing.read_text())
            logger.info("Pre-existing results loaded for %s: AUROC=%.4f", m, all_metrics[m]["AUROC"])

    for model_name in models:
        metrics_path = output_dir / model_name / f"{model_name}_metrics.json"
        if args.skip_existing and metrics_path.exists():
            logger.info("Skipping %s (already exists)", model_name)
            continue
        try:
            m = run_one(model_name, train_df, valid_df, test_df, vocab, output_dir, args.seed)
            all_metrics[model_name] = m
        except Exception as exc:
            logger.error("Model %s FAILED: %s", model_name, exc, exc_info=True)

    if all_metrics:
        build_combined_summary(all_metrics, output_dir)

    logger.info("All models complete.")


if __name__ == "__main__":
    main()
