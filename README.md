# MacaBrainNet v2

Monkey brain MRI segmentation pipeline: **skull stripping → tissue segmentation**.

Built with SwinUNETR (MONAI backend), PyTorch DDP training, and ensemble inference across 5-fold cross-validation models.

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended, CPU inference supported but slow)
- PyTorch 2.x
- MONAI (for SwinUNETR blocks only)
- nibabel, numpy, scipy

## Installation

```bash
# Clone the repository
git clone <repo-url> macaBrainNet_v2
cd macaBrainNet_v2

# Install dependencies
pip install torch monai nibabel scipy tqdm tensorboard

# Download pretrained model checkpoints from HuggingFace Hub
python download_from_hf.py
```

Model checkpoints will be saved to `swinunetr_models/` with this structure:

```
swinunetr_models/
├── skull_stripping/
│   ├── fold_1/best_3d_swinunetr_model.pth
│   ├── fold_2/best_3d_swinunetr_model.pth
│   ├── fold_3/best_3d_swinunetr_model.pth
│   ├── fold_4/best_3d_swinunetr_model.pth
│   └── fold_5/best_3d_swinunetr_model.pth
└── tissue_segmentation/
    ├── fold_1/best_3d_swinunetr_model.pth
    ├── fold_2/best_3d_swinunetr_model.pth
    ├── fold_3/best_3d_swinunetr_model.pth
    ├── fold_4/best_3d_swinunetr_model.pth
    └── fold_5/best_3d_swinunetr_model.pth
```

## Quick Start

```bash
# Run the full pipeline on example data
bash src/run_example.sh
```

This runs skull stripping + tissue segmentation on the included example T1w image.
Results are saved to `example_output/`.

## Pipeline Usage

The full pipeline runs four steps automatically:

```
Input T1w.nii.gz
    │
    ▼
Step 1: Skull Stripping (ensemble, 5-fold)
    ├── Model: SwinUNETR, 0.5mm, binary
    └── Output: *_brain_mask.nii.gz
    │
    ▼
Step 2: Brain BBox Crop + Brain Mask
    ├── Finds brain bounding box from mask
    ├── Expands by 16 voxels outward
    ├── Crops original image to bbox region
    ├── Applies brain mask (non-brain → min intensity)
    └── Output: *_cropped.nii.gz
    │
    ▼
Step 3: Tissue Segmentation (ensemble, 5-fold)
    ├── Model: SwinUNETR, 0.4mm, 19-class
    └── Output: *_tissue_seg_cropped.nii.gz (cropped space)
    │
    ▼
Step 4: Remap to Original Space
    ├── Places segmentation back into full image volume
    └── Output: *_tissue_seg.nii.gz (original space)
```

### Run from command line

```bash
python pipeline.py \
    --img /path/to/T1w.nii.gz \
    --out-dir ./output \
    --padding 16 \
    --device cuda
```

### Run from Python

```python
from pipeline import run_pipeline

run_pipeline(
    img_path="T1w.nii.gz",
    out_dir="./output",
    padding=16,
    device="cuda",
)
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--img` | (required) | Input T1w NIfTI image |
| `--out-dir` | (required) | Output directory |
| `--skull-ckpt-dir` | `swinunetr_models/skull_stripping` | Skull stripping checkpoints |
| `--tissue-ckpt-dir` | `swinunetr_models/tissue_segmentation` | Tissue seg checkpoints |
| `--skull-spacing` | 0.5 0.5 0.5 | Target spacing for skull strip (mm) |
| `--tissue-spacing` | 0.4 0.4 0.4 | Target spacing for tissue seg (mm) |
| `--patch-size` | 96 96 96 | Sliding window patch size |
| `--overlap` | 0.60 | Sliding window overlap ratio |
| `--padding` | 16 | Voxels to pad outward from brain bbox |
| `--device` | cuda | Device for inference |

## Single Model Inference

If you only need one task, you can use the individual prediction scripts:

```bash
# Skull stripping only (single checkpoint)
python predict.py \
    --img T1w.nii.gz \
    --out brain_mask.nii.gz \
    --ckpt swinunetr_models/skull_stripping/fold_1/best_3d_swinunetr_model.pth \
    --num-classes 2 \
    --spacing 0.5 0.5 0.5

# Skull stripping (ensemble, 5-fold averaging)
python predict_ensemble.py \
    --img T1w.nii.gz \
    --out brain_mask.nii.gz \
    --ckpt-dir swinunetr_models/skull_stripping \
    --num-classes 2 \
    --spacing 0.5 0.5 0.5

# Tissue segmentation (ensemble, 5-fold averaging)
python predict_ensemble.py \
    --img T1w_cropped.nii.gz \
    --out tissue_seg.nii.gz \
    --ckpt-dir swinunetr_models/tissue_segmentation \
    --num-classes 19 \
    --spacing 0.4 0.4 0.4
```

## Tissue Labels

The tissue segmentation model outputs 19 classes (0 = background):

| ID | Tissue | ID | Tissue |
|---|---|---|---|
| 0 | Background | 10 | Caudate |
| 1 | White Matter (WM) | 11 | Globus Pallidus |
| 2 | Cortex | 12 | Thalamus |
| 3 | Lateral Ventricle (LV) | 13 | Hippocampus |
| 4 | Cerebellum WM | 14 | Amygdala |
| 5 | Cerebellum Cortex | 15 | Brainstem |
| 6 | 3rd Ventricle | 16 | Corpus Callosum |
| 7 | 4th Ventricle | 17 | Optic Chiasm |
| 8 | Putamen | 18 | Pineal Gland |
| 9 | Accumbens | | |

## Training

To train your own models from scratch:

```bash
# Skull stripping (5-fold CV, 0.5mm, binary)
FOLD=1 bash train_skullstrip.sh

# Tissue segmentation (5-fold CV, 0.4mm, 19-class)
FOLD=1 bash train_tissueseg.sh
```

Training uses PyTorch DDP (default 4 GPUs). Configure via environment variables:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  \
DATA_ROOT=/path/to/data         \
FOLD=1                           \
bash train_skullstrip.sh
```

## Model Card

- **Architecture**: SwinUNETR v2 (MONAI)
  - `feature_size=48`, `patch_size=(2,2,2)`
  - 4 encoder stages: depths=[2,2,2,2], heads=[3,6,12,24]
  - Deep supervision: 5 output heads
- **Pretrained weights**: LocalGlobal (self-supervised, 17000 steps)
- **Skull stripping**: DiceFocalLoss, 0.5mm, patch 96³
- **Tissue segmentation**: Weighted DiceFocalLoss, 0.4mm, patch 96³
- **Ensemble**: Softmax averaging across 5 folds

## Upload Models to HuggingFace

```bash
python upload_to_hf.py
```

## Project Structure

```
macaBrainNet_v2/
├── nets/swinunetr.py          # SwinUNETR architecture
├── data/
│   ├── dataset.py             # NiftiPatchDataset + dataloaders
│   ├── transforms.py          # I/O, preprocessing, augmentations
│   └── utils.py               # Label encoding utilities
├── losses/dice_focal_loss.py  # DiceLoss, FocalLoss, DiceFocalLoss
├── trainer.py                 # DDP trainer + sliding_window_inference
├── train_skullstrip.py        # Skull stripping training
├── train_skullstrip.sh        # Training launcher
├── train_tissueseg.py         # Tissue segmentation training
├── train_tissueseg.sh         # Training launcher
├── predict.py                 # Single-checkpoint inference
├── predict_ensemble.py        # Ensemble inference (5-fold)
├── pipeline.py                # End-to-end pipeline
├── upload_to_hf.py            # Upload models to HuggingFace
├── download_from_hf.py        # Download models from HuggingFace
├── swinunetr_models/          # Model checkpoints
├── src/example/               # Example data
└── src/run_example.sh         # Example pipeline run
```
