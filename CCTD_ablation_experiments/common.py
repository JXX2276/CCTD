"""Shared utilities for the CCTD ablation experiments.

The ablations live under the cleaned main implementation and reuse its
prepared datasets. Run the scripts in this directory directly; their paths are
independent of the current working directory.
"""

import csv
import json
import sys
from pathlib import Path

import torch

MAIN_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
if str(MAIN_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_EXPERIMENT_DIR))

from CCTD import HSIDenoisingPipeline, load_and_preprocess_data  # noqa: E402
from utils import set_all_seeds  # noqa: E402


class NamedAblationPipeline(HSIDenoisingPipeline):
    """Base pipeline that keeps results from every ablation variant separate."""

    def _create_output_directory(self):
        self.output_dir = (
            Path(self.config["output_root"])
            / self.config["experiment_name"]
            / self.config["data_name"]
            / f"case{self.config['noise_case']}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)


def base_config(args, experiment_name):
    """Article protocol: Beijing, MSI guidance, and complex noise Case 6."""
    return {
        "experiment_name": experiment_name,
        "data_name": args.data_name,
        "gde_key": args.gde_key,
        "hsi_key": "hsi",
        "noise_case": args.noise_case,
        "omega": args.omega,
        "lr": args.lr,
        "max_iter": args.max_iter,
        "weight_recon": 1.0,
        "weight_guide": 200.0,
        "weight_nn": 90.0,
        "seed": args.seed,
        "eval_interval": args.eval_interval,
        "output_root": args.output_root,
    }


def run_pipeline(pipeline_class, config):
    """Load one prepared dataset, train, save outputs, and return final metrics."""
    set_all_seeds(config["seed"])
    data_path = MAIN_EXPERIMENT_DIR / "data" / f"{config['data_name']}.mat"
    ground_truth, guide, noisy_hsi = load_and_preprocess_data(
        data_path, config["noise_case"], config["hsi_key"], config["gde_key"]
    )
    pipeline = pipeline_class(config)
    pipeline.run_pipeline(ground_truth, guide, noisy_hsi)
    return {name: float(value) for name, value in pipeline.final_metrics.items()}


def run_pipeline_with_guide(pipeline_class, config, guide_transform):
    """As ``run_pipeline``, but transforms the loaded guide before training."""
    set_all_seeds(config["seed"])
    data_path = MAIN_EXPERIMENT_DIR / "data" / f"{config['data_name']}.mat"
    ground_truth, guide, noisy_hsi = load_and_preprocess_data(
        data_path, config["noise_case"], config["hsi_key"], config["gde_key"]
    )
    transformed_guide = guide_transform(guide)
    pipeline = pipeline_class(config)
    pipeline.run_pipeline(ground_truth, transformed_guide, noisy_hsi)
    return {name: float(value) for name, value in pipeline.final_metrics.items()}, guide, transformed_guide


def save_summary(rows, output_root, filename):
    """Write machine-readable summaries in JSON and CSV formats."""
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{filename}.json"
    csv_path = output_dir / f"{filename}.csv"

    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary to {json_path}")
    print(f"Saved summary to {csv_path}")


def cuda_available():
    return torch.cuda.is_available()
