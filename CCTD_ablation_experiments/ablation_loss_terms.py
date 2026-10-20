"""Reproduce the loss-term and spatial-coupling ablation on Beijing Case 6."""

import argparse
from pathlib import Path

import torch

from ablation_models import UncoupledSpatialCTN
from common import NamedAblationPipeline, base_config, run_pipeline, save_summary


class UncoupledSpatialPipeline(NamedAblationPipeline):
    def setup_network(self, hsi_shape, guide_shape):
        self.net = UncoupledSpatialCTN(
            hsi_shape, guide_shape, guide_shape[:2], hsi_shape[-1], omega=self.config["omega"], device=self.device
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config["lr"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_name", default="Beijing", choices=["Beijing", "Wuhan", "Florence"])
    parser.add_argument("--gde_key", default="msi", choices=["msi", "sar", "pan"])
    parser.add_argument("--noise_case", default=6, type=int)
    parser.add_argument("--max_iter", default=5001, type=int)
    parser.add_argument("--omega", default=3.0, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--eval_interval", default=100, type=int)
    parser.add_argument("--output_root", default="output_ablation")
    return parser.parse_args()


def main():
    args = parse_args()
    # (a)-(e) are the five settings reported in the paper's loss/coupling table.
    variants = [
        ("a_no_low_rank", NamedAblationPipeline, 1.0, 200.0, 0.0),
        ("b_reconstruction_only", NamedAblationPipeline, 1.0, 0.0, 0.0),
        ("c_reconstruction_low_rank", NamedAblationPipeline, 1.0, 0.0, 90.0),
        ("d_uncoupled_spatial", UncoupledSpatialPipeline, 1.0, 200.0, 90.0),
        ("e_full_cctd", NamedAblationPipeline, 1.0, 200.0, 90.0),
    ]
    rows = []
    for name, pipeline_class, recon_weight, guide_weight, low_rank_weight in variants:
        config = base_config(args, name)
        config.update(weight_recon=recon_weight, weight_guide=guide_weight, weight_nn=low_rank_weight)
        print(f"Running {name}")
        metrics = run_pipeline(pipeline_class, config)
        rows.append({"variant": name, **metrics})
    save_summary(rows, args.output_root, "loss_terms_summary")


if __name__ == "__main__":
    main()
