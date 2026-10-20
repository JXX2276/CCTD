"""Reproduce tensor-nuclear-norm and spectral-coupling modeling ablations."""

import argparse

import torch

from ablation_models import SharedSpectralCTN
from common import NamedAblationPipeline, base_config, run_pipeline, save_summary


class OverlapTensorNuclearNormPipeline(NamedAblationPipeline):
    def _overlap_tensor_nuclear_norm(self, tensor):
        alpha_h, alpha_w, alpha_b = self.config["tnn_alpha"]
        mode_h = tensor.reshape(tensor.shape[0], -1)
        mode_w = tensor.permute(1, 0, 2).reshape(tensor.shape[1], -1)
        mode_b = tensor.permute(2, 0, 1).reshape(tensor.shape[2], -1)
        return alpha_h * torch.norm(mode_h, "nuc") + alpha_w * torch.norm(mode_w, "nuc") + alpha_b * torch.norm(mode_b, "nuc")

    def _compute_loss(self, out_HSI, out_Guide, X, Guide):
        mask = X != 0
        recon_loss = torch.norm(out_HSI[mask] - X[mask], 1)
        guide_loss = torch.norm(out_Guide - Guide, 2)
        low_rank_loss = self._overlap_tensor_nuclear_norm(out_HSI)
        return (
            self.config["weight_recon"] * recon_loss
            + self.config["weight_guide"] * guide_loss
            + self.config["weight_nn"] * low_rank_loss
        )


class SharedSpectralPipeline(NamedAblationPipeline):
    def setup_network(self, hsi_shape, guide_shape):
        self.net = SharedSpectralCTN(
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
    parser.add_argument("--tnn_alpha", nargs=3, type=float, default=[0.05, 0.05, 1.0])
    return parser.parse_args()


def main():
    args = parse_args()
    variants = [
        ("a_tensor_nuclear_norm", OverlapTensorNuclearNormPipeline),
        ("b_shared_spectral", SharedSpectralPipeline),
        ("c_full_cctd", NamedAblationPipeline),
    ]
    rows = []
    for name, pipeline_class in variants:
        config = base_config(args, name)
        config["tnn_alpha"] = tuple(args.tnn_alpha)
        print(f"Running {name}")
        metrics = run_pipeline(pipeline_class, config)
        rows.append({"variant": name, **metrics})
    save_summary(rows, args.output_root, "modeling_strategies_summary")


if __name__ == "__main__":
    main()
