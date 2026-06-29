#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.data.immunogenicity import ImmunogenicityDataset
from benchmark.evaluate import GLYCANML_BASELINES, rank_against_baselines
from benchmark.models.glycanaa_trainer import GlycanAATrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


MODEL_NAME = "GlycanAA"


def _write_outputs(output_dir: Path, metrics: dict[str, float], ranking_df: pd.DataFrame, history: dict[str, list[float]], y_true, y_prob) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{MODEL_NAME}_metrics.json").write_text(json.dumps(metrics, indent=2))
    ranking_df.to_csv(output_dir / f"{MODEL_NAME}_ranking.csv")
    (output_dir / f"{MODEL_NAME}_history.json").write_text(json.dumps(history, indent=2))
    np.savez(output_dir / f"{MODEL_NAME}_arrays.npz", y_true=y_true, y_prob=y_prob)
    (output_dir / f"{MODEL_NAME}_summary.txt").write_text(_build_summary(metrics, ranking_df))


def _build_summary(metrics: dict[str, float], ranking_df: pd.DataFrame) -> str:
    our_row = ranking_df[ranking_df["Model"] == MODEL_NAME].iloc[0]
    rank = int(our_row.name)
    total = len(ranking_df)
    best_paper = ranking_df[ranking_df["Source"] == "GlycanML (paper)"]["AUROC"].max()
    lines = [
        "=" * 60,
        "GLYCAN IMMUNOGENICITY BENCHMARK — RESULTS",
        f"Run: {MODEL_NAME}",
        "=" * 60,
        "",
        f"AUROC  : {metrics['AUROC']}",
        f"AUPRC  : {metrics['AUPRC']}",
        f"Accuracy: {metrics['Accuracy']}",
        f"F1     : {metrics['F1']}",
        f"MCC    : {metrics['MCC']}",
        "",
        f"RANKING vs GlycanML paper baselines: #{rank} / {total}",
        f"vs best paper model: {metrics['AUROC'] - best_paper:+.4f} AUROC",
        "",
        ranking_df.to_string(),
    ]
    summary = "\n".join(lines)
    print(summary)
    return summary


def _update_combined_summary(metrics: dict[str, float], repo_root: Path) -> None:
    ranking_path = repo_root / "results" / "all_models_ranking.csv"
    summary_path = repo_root / "results" / "all_models_summary.txt"

    rows = []
    if ranking_path.exists():
        existing = pd.read_csv(ranking_path)
        if "Rank" in existing.columns:
            existing = existing.drop(columns=["Rank"])
        existing = existing[~((existing["Model"] == MODEL_NAME) & (existing["Source"] == "this benchmark"))]
        rows.extend(existing.to_dict("records"))
    else:
        for name, vals in GLYCANML_BASELINES.items():
            rows.append({"Model": name, "Source": "GlycanML (paper)", **vals, "F1": np.nan, "MCC": np.nan})

    rows.append({
        "Model": MODEL_NAME,
        "Source": "this benchmark",
        "AUROC": metrics["AUROC"],
        "AUPRC": metrics["AUPRC"],
        "F1": metrics["F1"],
        "MCC": metrics["MCC"],
    })

    df = pd.DataFrame(rows)
    for col in ["F1", "MCC"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df.sort_values("AUROC", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"
    df.to_csv(ranking_path)

    models_trained = ", ".join(df[df["Source"] == "this benchmark"]["Model"].tolist())
    lines = [
        "=" * 70,
        "GLYCAN IMMUNOGENICITY BENCHMARK — ALL MODELS SUMMARY",
        "=" * 70,
        "",
        df.to_string(),
        "",
        "Reference: Xu et al. 2024, arXiv:2405.16206",
        f"Models trained: {models_trained}",
    ]
    summary_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run torchdrug-free GlycanAA benchmark")
    parser.add_argument("--output-dir", default="results/GlycanAA_reimpl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    repo_root = Path(__file__).resolve().parents[1]

    logger.info("Loading immunogenicity dataset …")
    ds = ImmunogenicityDataset()
    logger.info(ds.stats())
    train_df, valid_df, test_df = ds.split()

    trainer = GlycanAATrainer(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )
    trainer.prepare_vocab(ds.df["glycan"].tolist())
    logger.info("Monosaccharide vocab size: %d", len(trainer.mono_vocab or {}))

    history = trainer.fit(train_df, valid_df)
    metrics, y_true, y_prob = trainer.evaluate(test_df)
    ranking_df = rank_against_baselines(metrics, model_label=MODEL_NAME)
    _write_outputs(output_dir, metrics, ranking_df, history, y_true, y_prob)
    trainer.save(output_dir / "model.pt")
    _update_combined_summary(metrics, repo_root)
    logger.info("Done. %s AUROC=%.4f AUPRC=%.4f", MODEL_NAME, metrics["AUROC"], metrics["AUPRC"])


if __name__ == "__main__":
    main()
