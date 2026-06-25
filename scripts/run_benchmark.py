#!/usr/bin/env python3
"""
CLI entry point for the glycan immunogenicity benchmark.

Examples
--------
# Run with defaults (downloads dataset automatically):
    python scripts/run_benchmark.py

# Custom output directory and more epochs:
    python scripts/run_benchmark.py --output-dir my_results --max-epochs 100

# Use a local dataset CSV:
    python scripts/run_benchmark.py --csv-path /path/to/glycan_immunogenicity.csv

# Use a config file:
    python scripts/run_benchmark.py --config configs/immunogenicity.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Pin to GPU 0 — must be set before any torch import
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from benchmark.pipeline import ImmunogenicityBenchmark


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Glycan immunogenicity benchmark (GlycanML / SweetNet)"
    )
    p.add_argument("--config", default="configs/immunogenicity.yaml",
                   help="YAML config file (default: configs/immunogenicity.yaml)")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory from config")
    p.add_argument("--csv-path", default=None,
                   help="Local CSV path (skips download if provided)")
    p.add_argument("--run-name", default=None,
                   help="Label for this run in result files")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    cfg: dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        logging.warning("Config file not found: %s — using defaults", config_path)

    model_cfg = cfg.get("model", {})
    out_cfg   = cfg.get("output", {})

    bench = ImmunogenicityBenchmark(
        output_dir  = args.output_dir  or out_cfg.get("dir", "results"),
        csv_path    = args.csv_path,
        lr          = args.lr          or model_cfg.get("lr",           1e-3),
        weight_decay=                     model_cfg.get("weight_decay", 1e-4),
        max_epochs  = args.max_epochs  or model_cfg.get("max_epochs",     50),
        batch_size  =                     model_cfg.get("batch_size",     64),
        patience    =                     model_cfg.get("patience",       10),
        seed        = args.seed        or model_cfg.get("seed",           42),
        run_name    = args.run_name    or out_cfg.get("run_name", "sweetnet"),
    )

    metrics = bench.run()

    print("\n✓ Benchmark complete.")
    print(f"  AUROC : {metrics['AUROC']}")
    print(f"  AUPRC : {metrics['AUPRC']}")
    print(f"  F1    : {metrics['F1']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
