# URRL

Official PyTorch implementation of **Uncertainty-Aware Reliable Region Learning (URRL)** for semi-supervised superficial-organ ultrasound image segmentation.

URRL uses two independently initialized U-Net branches and contains three training-only components:

- Uncertainty-Ranked Contrastive Learning (URC)
- Reliable Center Compactness (RCC)
- Uncertainty-Guided Region Exchange (URE)

The auxiliary components are removed at inference time, so testing uses one ordinary U-Net branch.

## Repository contents

```text
URRL/
├── train.py                 # CPS + URC + RCC + URE training
├── test.py                  # DSC, JC, 95HD, and ASD evaluation
├── smoke_test.py            # CPU network-shape check
├── dataloaders/dataset.py   # dataset loading and two-stream sampler
├── networks/                # U-Net backbone
├── utils/                   # losses and ramp-up functions
├── augmentations/           # CTAugment utilities
└── val_2D.py                # validation during training
```

## Installation

Python 3.9 or newer is recommended. Install a PyTorch build compatible with your CUDA version, then install the remaining dependencies:

```bash
conda create -n urrl python=3.12 -y
conda activate urrl
pip install -r requirements.txt
```

The released code was checked in an environment with Python 3.12.3, PyTorch 2.12.1, TorchVision 0.27.1, and CUDA 13.0.

Run the CPU smoke test after installation:

```bash
python smoke_test.py
```

## Data preparation

Download BUSI and TN3K from their official sources. The datasets are not redistributed in this repository. Organize each preprocessed dataset as follows; image and mask filenames must match.

```text
DATA_ROOT/
├── TrainDataset/
│   ├── images/              # TN3K may use trainval-images/
│   └── masks/               # TN3K may use trainval-masks/
└── TestDataset/
    ├── images/              # TN3K may use test-images/
    └── masks/               # TN3K may use test-masks/
```

Folder names only need to end in `TrainDataset`, `TestDataset`, `images`, or `masks`, so the TN3K names above are accepted. Images and masks are resized to 256 x 256. Masks are binarized at intensity 128.

For the training split, every fifth sorted sample is reserved for validation. The labeled subset is then sampled deterministically with seed 1337.

## Training

Example command for training with 10% labeled data, seed 1337, and 256 x 256 inputs:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --root_path /path/to/DATA_ROOT \
  --snapshot_path runs/urrl_10percent \
  --labeled_num 0.10 \
  --batch_size 16 \
  --labeled_bs 8 \
  --max_iterations 30000 \
  --seed 1337 \
  --urc_patch_size 8 \
  --urc_compact_weight 0.10 \
  --ure_weight 0.10 \
  --ure_block_size 8
```

For the 5% setting, change `--labeled_num 0.10` to `--labeled_num 0.05`. Training writes logs, TensorBoard events, and the best checkpoints for both branches to `--snapshot_path`.

## Evaluation

Evaluate either saved branch with:

```bash
python test.py \
  --root_path /path/to/DATA_ROOT \
  --ckpt_path runs/urrl_10percent/unet_best_model1.pth \
  --save_dir test_results/urrl_10percent
```

The evaluator writes `test_metrics.json` with per-case and mean DSC, JC, 95HD, and ASD, and saves up to 50 qualitative overlays in `vis/`. Use `unet_best_model2.pth` to evaluate the second branch.

## Reproducibility notes

- Default random seed: 1337
- Input size: 256 x 256
- Number of classes: 2 (background and lesion)
- Optimizer: SGD with learning rate 0.02, momentum 0.9, and weight decay 0.0001
- Training length: 30,000 iterations
- Validation and testing use a single segmentation branch

## License

This repository follows the MIT license included in [LICENSE](LICENSE). The U-Net training framework is derived from SSL4MIS; retain the original license notice when redistributing the code.
