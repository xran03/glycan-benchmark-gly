"""
Standalone script: re-generate all plots from saved benchmark outputs.

Usage (after a training run):
    python results/plot/regenerate_plots.py --results-dir results/ --run-name sweetnet
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description="Re-generate benchmark plots from saved outputs")
    p.add_argument("--results-dir", default="results", help="Directory with saved outputs")
    p.add_argument("--run-name", default="sweetnet")
    p.add_argument("--plot-dir", default=None, help="Override plot output dir")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    plot_dir = Path(args.plot_dir) if args.plot_dir else results_dir / "plot"

    metrics_path = results_dir / f"{args.run_name}_metrics.json"
    arrays_path  = results_dir / f"{args.run_name}_arrays.npz"
    history_path = results_dir / f"{args.run_name}_history.json"

    if not metrics_path.exists():
        logger.error("Metrics file not found: %s", metrics_path)
        sys.exit(1)

    metrics = json.loads(metrics_path.read_text())
    logger.info("Loaded metrics: %s", metrics)

    history: dict = {}
    if history_path.exists():
        history = json.loads(history_path.read_text())
    else:
        logger.warning("History file not found — training curve will be skipped")

    y_true = y_prob = None
    if arrays_path.exists():
        data   = np.load(arrays_path)
        y_true = data["y_true"]
        y_prob = data["y_prob"]
    else:
        logger.warning("Arrays file not found — ROC/PR/confusion plots will be skipped")

    from results.plot.plot_results import (
        plot_baseline_ranking,
        plot_confusion_matrix,
        plot_pr,
        plot_roc,
        plot_training_loss,
    )

    if y_true is not None and y_prob is not None:
        plot_roc(y_true, y_prob, metrics["AUROC"], plot_dir)
        plot_pr(y_true, y_prob, metrics["AUPRC"], plot_dir)
        plot_confusion_matrix(y_true, y_prob, plot_dir)
    if history:
        plot_training_loss(history, plot_dir)
    plot_baseline_ranking(metrics, plot_dir)

    logger.info("Done — plots saved to %s/", plot_dir)


if __name__ == "__main__":
    main()
