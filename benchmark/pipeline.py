"""
Main orchestrator for the glycan immunogenicity benchmark.

Usage (Python API):
    from benchmark.pipeline import ImmunogenicityBenchmark
    bench = ImmunogenicityBenchmark(output_dir="results/")
    bench.run()

Usage (CLI):
    see scripts/run_benchmark.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from benchmark.data.immunogenicity import ImmunogenicityDataset
from benchmark.evaluate import compute_metrics, rank_against_baselines, save_results
from benchmark.models.sweetnet_classifier import SweetNetClassifier

logger = logging.getLogger(__name__)


class ImmunogenicityBenchmark:
    """
    End-to-end benchmark pipeline:
      1. Download / load dataset
      2. Encode glycans with pretrained SweetNet
      3. Fine-tune MLP classification head
      4. Evaluate on held-out test set
      5. Rank against GlycanML paper baselines
      6. Save results to output_dir
    """

    def __init__(
        self,
        output_dir: str | Path = "results",
        csv_path: str | Path | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        seed: int = 42,
        run_name: str = "sweetnet",
    ):
        self.output_dir = Path(output_dir)
        self.csv_path = Path(csv_path) if csv_path else None
        self.run_name = run_name
        self.model_kwargs = dict(
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            seed=seed,
        )

    def run(self) -> dict[str, float]:
        """
        Execute the full pipeline.

        Returns:
            metrics dict with AUROC, AUPRC, Accuracy, F1, MCC on the test set.
        """
        # 1. Load data
        logger.info("Loading immunogenicity dataset …")
        dataset = ImmunogenicityDataset(self.csv_path)
        logger.info(dataset.stats())
        train_df, valid_df, test_df = dataset.split()
        vocab = dataset.vocab

        # 2 + 3. Train
        logger.info("Training SweetNet classifier (vocab_size=%d) …", len(vocab))
        model = SweetNetClassifier(vocab=vocab, **self.model_kwargs)
        history = model.fit(train_df, valid_df)

        # 4. Evaluate on test
        logger.info("Evaluating on test set …")
        y_prob = model.predict_proba(test_df["glycan"].tolist())
        y_true = test_df["label"].values
        metrics = compute_metrics(y_true, y_prob)

        # 5. Rank
        ranking_df = rank_against_baselines(metrics)

        # 6. Save metrics, arrays, history
        save_results(metrics, ranking_df, self.output_dir, run_name=self.run_name)
        np.savez(
            self.output_dir / f"{self.run_name}_arrays.npz",
            y_true=y_true,
            y_prob=y_prob,
        )
        (self.output_dir / f"{self.run_name}_history.json").write_text(
            json.dumps(history, indent=2)
        )

        # 7. Generate plots
        plot_dir = self.output_dir / "plot"
        try:
            from results.plot.plot_results import plot_all
            plot_all(y_true, y_prob, history, metrics, output_dir=plot_dir)
        except Exception as exc:
            logger.warning("Plotting failed (non-fatal): %s", exc)

        # 8. Save full model weights
        weights_path = self.output_dir / f"{self.run_name}_model.pt"
        model.save(weights_path)

        return metrics
