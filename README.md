# GAN-based augmentation and SSIM-guided selection for breast ultrasound classification

Code for the manuscript "[paper title]". It covers GAN training and image generation,
SSIM-based selection of synthetic images, CNN training/evaluation on an internal and an
external test set, and the robustness analyses (conventional augmentation, Top-k, Random-5,
SSIM thresholds, three random seeds, bootstrap confidence intervals).

## Data
Both datasets are public and are not included here:
- BUSI (primary dataset): Al-Dhabyani et al. [11]
- External test set: Yap et al. [12]

Expected layout:
```
<BASE>/split_dataset/train/{benign,malignant}
<BASE>/split_dataset/test/{benign,malignant}     # internal test
<BASE>/test1/{b,m}                               # external test
```
All scripts use a `BASE` path at the top of the file (`F:\jihad\Breast GAN types\breast\breast`).
Change it to your own folder before running.

## Installation
```
pip install -r requirements.txt
```

## Pipeline
| Step | Script | Paper section |
|---|---|---|
| 1. Train GANs and generate 10 images per original | `gan/pix2pix.py`, `gan/cycleGAN.py`, `gan/WGAN.py`, `gan/DCGAN.py`, `gan/CGAN.py` | 3.3 |
| 2. Top-5 selection by source-image SSIM | `selection/select_top5.py` | 3.4, 3.5.2 |
| 3. Top-3 / Top-10 / Random-5 subsets | `selection/make_subsets.py --gan pix2pix` (and `--gan cyclegan`) | 3.5.4 |
| 4. SSIM threshold sensitivity | `selection/ssim_threshold_all.py` | 3.5.5, 4.7 |
| 5. CNN training: conventional augmentation, Top-k, Random-5, three seeds | `classification/run_all_50.py`, `classification/run_random5_seeds.py` (use `classification/train_conventional_aug.py`, `classification/run_seeds.py`) | 3.5, 4.4-4.6 |
| 6. External evaluation of a saved model | `classification/test_external.py` | 4 |
| 7. Bootstrap 95% CIs and paired tests | `analysis/bootstrap_ci.py --results_dir results_50ep --reference Orig_ConvAug` | 3.6 |
| 8. Pix2Pix with class-specific generators (malignant generator + all Pix2Pix runs) | `sensitivity/rerun_pix2pix_malignant.py` | 3.3.5, 4.3-4.7 |

Scripts in `classification/` and `sensitivity/` import each other; run them from one folder
that contains all of them.

## Note on Pix2Pix
In `gan/pix2pix.py`, `return models` sits inside the class loop of `train_pix2pix()`, so this
script trains only the benign generator and then applies it to both classes. The results
reported in the paper use separate benign and malignant generators: the malignant generator
and all Pix2Pix analyses are produced by `sensitivity/rerun_pix2pix_malignant.py` (the benign
generator from `gan/pix2pix.py` is reused). `gan/pix2pix.py` is kept unchanged so that the
earlier benign-only results (mentioned in the Limitations) can also be reproduced.

## Trained models
[Link to Zenodo / GitHub Release with the generator weights, if shared]

## Citation
[Citation of the paper]
