"""
Evaluation metrics for binary glycan immunogenicity classification.

Primary metrics (matching GlycanML paper):
  - AUROC  (higher is better)
  - AUPRC  (higher is better)

Secondary metrics:
  - Accuracy, F1, MCC at threshold 0.5
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GlycanML paper baselines (Xu et al. 2024, arXiv:2405.16206)
# ---------------------------------------------------------------------------
GLYCANML_PAPER_BASELINES: dict[str, dict[str, float]] = {
    "CNN":    {"AUROC": 0.741, "AUPRC": 0.639},
    "ResNet": {"AUROC": 0.756, "AUPRC": 0.654},
    "LSTM":   {"AUROC": 0.728, "AUPRC": 0.621},
    "BERT":   {"AUROC": 0.770, "AUPRC": 0.671},
    "GCN":    {"AUROC": 0.749, "AUPRC": 0.645},
    "GAT":    {"AUROC": 0.762, "AUPRC": 0.660},
    "GIN":    {"AUROC": 0.778, "AUPRC": 0.677},
    "MPNN":   {"AUROC": 0.785, "AUPRC": 0.684},
    "RGCN":   {"AUROC": 0.758, "AUPRC": 0.655},
    "CompGCN":{"AUROC": 0.752, "AUPRC": 0.649},
}

# ---------------------------------------------------------------------------
# Our trained baselines (same dataset/splits, GPU training on H200)
# ---------------------------------------------------------------------------
OUR_TRAINED_BASELINES: dict[str, dict[str, float]] = {
    "GCN":      {"AUROC": 0.9876, "AUPRC": 0.9194, "F1": 0.8000, "MCC": 0.7835},
    "LSTM":     {"AUROC": 0.9868, "AUPRC": 0.9238, "F1": 0.8000, "MCC": 0.7680},
    "MPNN":     {"AUROC": 0.9836, "AUPRC": 0.8770, "F1": 0.8293, "MCC": 0.8015},
    "GIN":      {"AUROC": 0.9808, "AUPRC": 0.8655, "F1": 0.8293, "MCC": 0.8015},
    "SweetNet": {"AUROC": 0.9792, "AUPRC": 0.8935, "F1": 0.8333, "MCC": 0.8166},
    "CNN":      {"AUROC": 0.9764, "AUPRC": 0.8818, "F1": 0.6667, "MCC": 0.6804},
    "GlycanAA": {"AUROC": 0.9708, "AUPRC": 0.8026, "F1": 0.5333, "MCC": 0.5226},
    "ResNet":   {"AUROC": 0.9544, "AUPRC": 0.7994, "F1": 0.5517, "MCC": 0.5602},
    "GAT":      {"AUROC": 0.9452, "AUPRC": 0.8507, "F1": 0.5185, "MCC": 0.5631},
}

# Default: rank against our own trained results (competitive baseline)
GLYCANML_BASELINES = OUR_TRAINED_BASELINES


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all evaluation metrics for a single run."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUROC":    round(float(roc_auc_score(y_true, y_prob)), 4),
        "AUPRC":    round(float(average_precision_score(y_true, y_prob)), 4),
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "F1":       round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "MCC":      round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


def rank_against_baselines(
    metrics: dict[str, float],
    model_label: str = "SweetNet (ours)",
) -> pd.DataFrame:
    """
    Return a DataFrame that places our model alongside GlycanML baselines,
    sorted by AUROC descending.
    """
    rows = []
    source_label = ("our trained baselines"
                    if GLYCANML_BASELINES is OUR_TRAINED_BASELINES
                    else "GlycanML (paper)")
    for model, vals in GLYCANML_BASELINES.items():
        rows.append({"Model": model, "Source": source_label, **vals})

    rows.append({
        "Model":  model_label,
        "Source": "this benchmark",
        "AUROC":  metrics.get("AUROC", float("nan")),
        "AUPRC":  metrics.get("AUPRC", float("nan")),
    })

    df = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-based rank
    df.index.name = "Rank"
    return df


def save_results(
    metrics: dict[str, float],
    ranking_df: pd.DataFrame,
    output_dir: Path | str,
    run_name: str = "sweetnet",
) -> None:
    """Persist metrics JSON + ranking CSV + plain-text summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    (out / f"{run_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    # Ranking CSV
    ranking_df.to_csv(out / f"{run_name}_ranking.csv")

    # Text summary
    summary = _build_summary(metrics, ranking_df, run_name)
    (out / f"{run_name}_summary.txt").write_text(summary)

    logger.info("Results saved to %s/", out)
    print(summary)


def _build_summary(
    metrics: dict[str, float],
    ranking_df: pd.DataFrame,
    run_name: str,
) -> str:
    our_row = ranking_df[ranking_df["Source"] == "this benchmark"]
    rank = our_row.index[0] if len(our_row) else "?"
    total = len(ranking_df)

    # Compare against the best trained baseline (not paper numbers)
    trained_rows = ranking_df[ranking_df["Source"] != "this benchmark"]
    best_baseline_auroc = trained_rows["AUROC"].max() if len(trained_rows) else float("nan")
    delta = metrics.get("AUROC", 0) - best_baseline_auroc
    baseline_label = ("our trained baselines"
                      if GLYCANML_BASELINES is OUR_TRAINED_BASELINES
                      else "GlycanML (paper)")

    lines = [
        "=" * 60,
        "GLYCAN IMMUNOGENICITY BENCHMARK — RESULTS",
        f"Run: {run_name}",
        "=" * 60,
        "",
        f"MODEL: {run_name}",
        f"  AUROC  : {metrics.get('AUROC', 'N/A')}",
        f"  AUPRC  : {metrics.get('AUPRC', 'N/A')}",
        f"  F1     : {metrics.get('F1', 'N/A')}",
        f"  MCC    : {metrics.get('MCC', 'N/A')}",
        "",
        f"RANKING vs {baseline_label}: #{rank} / {total}",
        f"  vs best baseline: {delta:+.4f} AUROC",
        "",
        "FULL RANKING",
        "-" * 60,
        ranking_df.to_string(),
        "",
        "Reference: Xu et al. 2024, arXiv:2405.16206",
        "Dataset  : GlycanML immunogenicity (Train 1026 / Valid 149 / Test 145)",
    ]
    return "\n".join(lines)
