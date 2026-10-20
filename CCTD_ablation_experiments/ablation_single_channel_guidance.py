"""Compare CCTD with a single-channel compressed version of its guide."""

import argparse

import numpy as np

from common import NamedAblationPipeline, base_config, run_pipeline_with_guide, save_summary


def compress_to_single_channel(guide):
    """Match the article's guide compression: channel-wise mean for MSI/SAR."""
    if guide.ndim == 2 or guide.shape[-1] == 1:
        return guide[..., None] if guide.ndim == 2 else guide
    return np.mean(guide, axis=-1, keepdims=True, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_name", default="Beijing", choices=["Beijing", "Wuhan"])
    parser.add_argument("--gde_key", default="msi", choices=["msi", "sar"])
    parser.add_argument("--noise_case", default="all", choices=["all", "1", "2", "3", "4", "5", "6"])
    parser.add_argument("--max_iter", default=5001, type=int)
    parser.add_argument("--omega", default=3.0, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--eval_interval", default=100, type=int)
    parser.add_argument("--output_root", default="output_ablation")
    return parser.parse_args()


def main():
    args = parse_args()
    cases = range(1, 7) if args.noise_case == "all" else [int(args.noise_case)]
    rows = []
    for noise_case in cases:
        args.noise_case = noise_case
        config = base_config(args, f"single_channel_case{noise_case}")
        metrics, original_guide, compressed_guide = run_pipeline_with_guide(
            NamedAblationPipeline, config, compress_to_single_channel
        )
        rows.append(
            {
                "noise_case": noise_case,
                "guide_channels_original": int(original_guide.shape[-1]),
                "guide_channels_used": int(compressed_guide.shape[-1]),
                **metrics,
            }
        )
    save_summary(rows, args.output_root, "single_channel_guidance_summary")


if __name__ == "__main__":
    main()
