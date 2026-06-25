"""
Plotting utilities for the glycan immunogenicity benchmark.

Generates and saves:
  - roc_curve.png        — ROC curve with AUROC annotation
  - pr_curve.png         — Precision-Recall curve with AUPRC annotation
  - training_loss.png    — train vs. val loss per epoch
  - baseline_ranking.png — bar chart comparing our model vs GlycanML baselines
  - confusion_matrix.png — confusion matrix at threshold 0.5

All functions accept an output_dir and save PNGs there.
Usage:
    from results.plot.plot_results import plot_all
    plot_all(y_true, y_prob, history, metrics, output_dir="results/plot")
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)

# ── shared style ────────────────────────────────────────────────────────────
PALETTE = {
    "ours":      "#E87722",   # Pfizer orange
    "seq_model": "#4C6EF5",   # indigo
    "gnn_model": "#37B24D",   # green
    "baseline":  "#868E96",   # grey
}
PLT_STYLE = {
    "figure.figsize":  (7, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
}


def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", path)


# ── individual plot functions ────────────────────────────────────────────────

def plot_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auroc: float,
    output_dir: Path,
) -> None:
    """ROC curve with diagonal reference and AUROC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color=PALETTE["ours"], lw=2,
                label=f"SweetNet  AUROC = {auroc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random  AUROC = 0.5")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Glycan Immunogenicity")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    _savefig(fig, output_dir / "roc_curve.png")


def plot_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auprc: float,
    output_dir: Path,
) -> None:
    """Precision-Recall curve with baseline (class imbalance) and AUPRC annotation."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    baseline = y_true.mean()
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color=PALETTE["ours"], lw=2,
                label=f"SweetNet  AUPRC = {auprc:.4f}")
        ax.axhline(baseline, color="k", lw=1, ls="--",
                   label=f"Random  AUPRC ≈ {baseline:.2f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve — Glycan Immunogenicity")
        ax.legend(loc="upper right")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    _savefig(fig, output_dir / "pr_curve.png")


def plot_training_loss(
    history: dict[str, list[float]],
    output_dir: Path,
) -> None:
    """Train vs. validation loss per epoch with early-stop marker."""
    train_loss = history.get("train_loss", [])
    val_loss   = history.get("val_loss", [])
    epochs = range(1, len(train_loss) + 1)

    best_ep = int(np.argmin(val_loss)) + 1 if val_loss else None

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots()
        ax.plot(epochs, train_loss, color=PALETTE["seq_model"], lw=2, label="Train loss")
        ax.plot(epochs, val_loss,   color=PALETTE["ours"],      lw=2, label="Val loss")
        if best_ep:
            ax.axvline(best_ep, color="grey", lw=1, ls=":", label=f"Best epoch ({best_ep})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.set_title("Training Curve — SweetNet MLP Head")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend()
    _savefig(fig, output_dir / "training_loss.png")


def plot_baseline_ranking(
    metrics: dict[str, float],
    output_dir: Path,
) -> None:
    """Horizontal bar chart: our model vs all GlycanML baselines, sorted by AUROC."""
    from benchmark.evaluate import GLYCANML_BASELINES  # local import to avoid circular

    rows = []
    for model, vals in GLYCANML_BASELINES.items():
        model_type = "seq_model" if model in {"CNN", "ResNet", "LSTM", "BERT"} else "gnn_model"
        rows.append({"Model": model, "AUROC": vals["AUROC"],
                     "AUPRC": vals["AUPRC"], "type": model_type})
    rows.append({"Model": "SweetNet\n(ours)", "AUROC": metrics["AUROC"],
                 "AUPRC": metrics["AUPRC"], "type": "ours"})

    df = pd.DataFrame(rows).sort_values("AUROC")

    colors = [PALETTE[t] for t in df["type"]]

    with plt.rc_context({**PLT_STYLE, "figure.figsize": (8, 6)}):
        fig, axes = plt.subplots(1, 2, sharey=True)
        for ax, col, title in zip(axes, ["AUROC", "AUPRC"], ["AUROC", "AUPRC"]):
            bars = ax.barh(df["Model"], df[col], color=colors, edgecolor="white", height=0.6)
            ax.set_xlabel(title)
            ax.set_title(title)
            ax.set_xlim(0.55, 0.85)
            # annotate values
            for bar, val in zip(bars, df[col]):
                ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=9)
        axes[0].set_ylabel("Model")
        fig.suptitle("Glycan Immunogenicity — Benchmark Ranking\nvs GlycanML Baselines (Xu et al. 2024)",
                     fontsize=12)
        # legend
        from matplotlib.patches import Patch
        legend_els = [
            Patch(color=PALETTE["ours"],      label="SweetNet (ours)"),
            Patch(color=PALETTE["gnn_model"], label="Graph models (paper)"),
            Patch(color=PALETTE["seq_model"], label="Sequence models (paper)"),
        ]
        fig.legend(handles=legend_els, loc="lower right", fontsize=9)
        fig.tight_layout()
    _savefig(fig, output_dir / "baseline_ranking.png")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    threshold: float = 0.5,
) -> None:
    """Confusion matrix at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(4, 4))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Non-immuno", "Immunogenic"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix (threshold={threshold})")
        fig.tight_layout()
    _savefig(fig, output_dir / "confusion_matrix.png")


# ── convenience wrapper ──────────────────────────────────────────────────────

def plot_all(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    history: dict[str, list[float]],
    metrics: dict[str, float],
    output_dir: str | Path = "results/plot",
) -> None:
    """
    Generate and save all benchmark plots.

    Args:
        y_true    : ground-truth labels (test set)
        y_prob    : predicted probabilities (test set)
        history   : {'train_loss': [...], 'val_loss': [...]}
        metrics   : {'AUROC': ..., 'AUPRC': ..., ...}
        output_dir: directory to write PNG files
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_roc(y_true, y_prob, metrics["AUROC"], out)
    plot_pr(y_true, y_prob, metrics["AUPRC"], out)
    plot_training_loss(history, out)
    plot_baseline_ranking(metrics, out)
    plot_confusion_matrix(y_true, y_prob, out)

    logger.info("All plots written to %s/", out)
