# Continuous Coupled Tucker Decomposition (CCTD)

Official PyTorch implementation for **Unified Guided Hyperspectral Image
Denoising by Continuous Coupled Tucker Decomposition**.

CCTD denoises a hyperspectral image (HSI) with a co-registered guidance
image. It shares the Tucker core and spatial implicit neural representations
between modalities, while keeping separate spectral factors. The objective
combines HSI reconstruction fidelity, guidance fidelity, and a spectral-mode
nuclear-norm regularizer.

## Requirements

- Python 3.9 or later
- PyTorch (CUDA is recommended)

Install the Python dependencies in your environment:

```bash
pip install torch numpy scipy scikit-image h5py
```

## Project structure

```text
CCTD_main_experiment/
|-- CCTD.py                         # Main experiment entry point
|-- network.py                      # CCTD network architecture
|-- data/                           # Prepared MATLAB datasets
|   |-- Beijing.mat                 # hsi, msi, sar
|   |-- Wuhan.mat                   # hsi, msi, sar
|   `-- Florence.mat                # hsi, pan
|-- utils/                          # Data loading, noise simulation, metrics
`-- CCTD_ablation_experiments/      # Paper ablation experiments
```

## Main experiment

Run the following command from this directory:

```bash
python CCTD.py
```

The default configuration evaluates Beijing and Wuhan with MSI/SAR guidance,
and Florence with PAN guidance, over all six simulated noise cases.

Results are saved under:

```text
output/<dataset>/case<noise_case>/
|-- data.mat              # Clean HSI, guide, and simulated noisy HSI
|-- CCTD_<guide>.mat      # Final reconstructed HSI
|-- metrics.json          # PSNR, SSIM, ERGAS, SAM, and elapsed time
`-- psnr_history.json     # PSNR logged during optimization
```

## Ablation experiments

The ablation scripts run the article's default Beijing/MSI/Case-6 protocol,
unless their command-line options override it:

```bash
cd CCTD_ablation_experiments

# HSI fidelity, guide fidelity, spectral low-rank prior, and spatial coupling
python ablation_loss_terms.py

# Overlap tensor nuclear norm, shared spectral INR, and full CCTD
python ablation_modeling_strategies.py

# Multi-channel versus channel-compressed guidance
python ablation_single_channel_guidance.py
```
## Citation
If you use this code in your research, please cite the corresponding paper:
```
@article{CCTD,
  title   = {Unified Guided Hyperspectral Image Denoising by Continuous Coupled Tucker Decomposition},
  author  = {Ji Xiaoxuan and
             Li Pengxian and
             Wang, Jialin and
             Xu Shuang and
             Ji Teng-Yu and
             Peng Jiangjun and
             Cao Xiangyong and
             Meng Deyu},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {64},
  year    = {2026},
  doi     = {10.1109/TGRS.2026.3708294},
  url     = {https://ieeexplore.ieee.org/document/11586087}
}
```
