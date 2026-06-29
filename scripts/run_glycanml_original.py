"""
Run the original GlycanML models using the torchdrug-based reference code.

This script uses the `tdrug` conda environment (Python 3.9 + torchdrug 0.2.1)
and the GlycanML_ref codebase to reproduce the paper's results.

Usage:
    /home/ranx03/miniconda3/envs/tdrug/bin/python \
        scripts/run_glycanml_original.py --models GCN,GAT,GIN

Output:
    results/glycanml_original/<model>/  — metrics JSON + summary
    results/glycanml_original_summary.csv
"""

import argparse
import json
import logging
import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GLYCANML_REF = Path("/home/ranx03/benchmark_gly/GlycanML_ref")
SCRATCH_DATA = Path("~/scratch/glycan-datasets").expanduser()
SCRATCH_OUTPUT = Path("~/scratch/torchglycan_output").expanduser()
RESULTS_DIR = Path("/home/ranx03/benchmark_gly/results/glycanml_original")

ALL_MODELS = ["GCN", "GAT", "GIN", "MPNN", "CNN", "ResNet", "LSTM"]

TDRUG_PYTHON = "/home/ranx03/miniconda3/envs/tdrug/bin/python"


def get_config_path(model: str) -> Path:
    return GLYCANML_REF / "configs" / "single_task" / model / f"immunogenicity_{model}.yaml"


def run_model(model: str, seed: int = 42, skip_existing: bool = False) -> dict | None:
    """Run a single GlycanML model and return metrics."""
    out_dir = RESULTS_DIR / model
    metrics_file = out_dir / f"{model}_glycanml_metrics.json"

    if skip_existing and metrics_file.exists():
        logger.info("[%s] Skipping — metrics already exist", model)
        return json.loads(metrics_file.read_text())

    config_path = get_config_path(model)
    if not config_path.exists():
        logger.warning("[%s] Config not found: %s", model, config_path)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    SCRATCH_DATA.mkdir(parents=True, exist_ok=True)
    SCRATCH_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Patch config: override output_dir + data path + gpus
    config_content = config_path.read_text()
    config_content = config_content.replace(
        "~/scratch/torchglycan_output/",
        str(SCRATCH_OUTPUT) + "/"
    ).replace(
        "~/scratch/glycan-datasets/",
        str(SCRATCH_DATA) + "/"
    ).replace(
        "gpus: {{ gpus }}",
        "gpus: [0]"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        tmp_config = f.name

    wrapper_script = f"""
import os, sys, json, math, shutil, random
import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "{GLYCANML_REF}")

from module import custom_data, custom_datasets, custom_models, custom_tasks, util

seed = {seed}
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

from easydict import EasyDict
import yaml
from jinja2 import Environment
from torchdrug import core

with open("{tmp_config}") as f:
    raw = f.read()
cfg = EasyDict(yaml.safe_load(raw))

# Build dataset
from module.custom_datasets.glycan_immunogenicity import GlycanImmunogenicityDataset
dataset = GlycanImmunogenicityDataset(path="{SCRATCH_DATA}")
train_set, valid_set, test_set = dataset.split()
print(f"Train: {{len(train_set)}}, Valid: {{len(valid_set)}}, Test: {{len(test_set)}}")

# Build model + task
task_cfg = cfg.task
task_cfg.task = cfg.dataset.target_fields
solver, scheduler = util.build_solver(cfg, dataset)

# Train
step = 10
best_result = float("-inf")
best_epoch = -1
work_dir = "{out_dir}"
os.chdir(work_dir)

for i in range(0, cfg.train.num_epoch, step):
    kwargs = {{"num_epoch": min(step, cfg.train.num_epoch - i)}}
    solver.train(**kwargs)
    epoch = solver.epoch
    solver.save(f"model_epoch_{{epoch}}.pth")
    metric = solver.evaluate("valid")
    result = metric[cfg.metric]
    print(f"Epoch {{epoch}}: valid {{cfg.metric}} = {{result:.4f}}")
    if result > best_result:
        best_result = result
        best_epoch = epoch
    torch.cuda.empty_cache()

# Load best model
shutil.move(f"model_epoch_{{best_epoch}}.pth", f"best_model_epoch_{{best_epoch}}.pth")
for fname in os.listdir("."):
    if fname.startswith("model_epoch_"):
        os.remove(fname)
solver.load(f"best_model_epoch_{{best_epoch}}.pth", load_optimizer=False)

# Evaluate on test
valid_metrics = solver.evaluate("valid")
test_metrics = solver.evaluate("test")
print("Valid metrics:", valid_metrics)
print("Test  metrics:", test_metrics)

# Save results
results = {{
    "model": "{model}",
    "seed": {seed},
    "best_valid_epoch": best_epoch,
    "valid": {{k: float(v) for k, v in valid_metrics.items()}},
    "test": {{k: float(v) for k, v in test_metrics.items()}},
    "AUROC": float(test_metrics.get("auroc [immunogenicity]", 0)),
    "AUPRC": float(test_metrics.get("auprc [immunogenicity]", 0)),
}}
with open("{metrics_file}", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to {metrics_file}")
"""

    wrapper_file = out_dir / "_run_wrapper.py"
    wrapper_file.write_text(wrapper_script)

    logger.info("[%s] Starting training (seed=%d) …", model, seed)
    try:
        result = subprocess.run(
            [TDRUG_PYTHON, str(wrapper_file)],
            capture_output=False,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            logger.error("[%s] Training failed (exit code %d)", model, result.returncode)
            return None
    except subprocess.TimeoutExpired:
        logger.error("[%s] Timed out after 3600s", model)
        return None
    finally:
        os.unlink(tmp_config)

    if metrics_file.exists():
        return json.loads(metrics_file.read_text())
    return None


def main():
    parser = argparse.ArgumentParser(description="Run original GlycanML models")
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model in models:
        metrics = run_model(model, seed=args.seed, skip_existing=args.skip_existing)
        if metrics:
            all_results[model] = metrics
            logger.info("[%s] AUROC=%.4f  AUPRC=%.4f",
                        model, metrics.get("AUROC", 0), metrics.get("AUPRC", 0))

    # Combined summary
    if all_results:
        import pandas as pd
        rows = []
        for m, r in sorted(all_results.items(), key=lambda x: -x[1].get("AUROC", 0)):
            rows.append({
                "Model": m,
                "AUROC": r.get("AUROC", 0),
                "AUPRC": r.get("AUPRC", 0),
                "Source": "GlycanML original (reproduced)",
            })
        df = pd.DataFrame(rows).set_index("Model")
        summary_file = RESULTS_DIR / "glycanml_original_summary.csv"
        df.to_csv(summary_file)
        print("\n" + "="*60)
        print("GlycanML ORIGINAL CODE — REPRODUCED RESULTS")
        print("="*60)
        print(df.to_string())
        print(f"\nSaved: {summary_file}")


if __name__ == "__main__":
    main()
