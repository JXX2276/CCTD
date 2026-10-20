# CCTD ablation experiments

These scripts reproduce the ablations reported for CCTD. They reuse the
prepared datasets and the main training pipeline in the parent directory; run
them from this directory or from any working directory.

| Script | Paper experiment | Default protocol |
| --- | --- | --- |
| `ablation_loss_terms.py` | HSI fidelity, guide fidelity, spectral nuclear norm, and spatial coupling | Beijing, MSI, Case 6 |
| `ablation_modeling_strategies.py` | Overlap tensor nuclear norm, shared spectral INR, and full CCTD | Beijing, MSI, Case 6 |
| `ablation_single_channel_guidance.py` | Multi-channel guide versus channel-compressed guide | Beijing, MSI, Cases 1-6 |

Run the default protocols with:

```powershell
python ablation_loss_terms.py
python ablation_modeling_strategies.py
python ablation_single_channel_guidance.py
```

Each method writes its reconstructed HSI (`.mat`), final metrics, and PSNR
history beneath `output_ablation/`. Each script also writes a JSON and CSV
summary in that directory. The default run length is 5001 iterations, matching
the main experiment; use `--max_iter 10` only as a smoke test.

No image visualization is generated.
