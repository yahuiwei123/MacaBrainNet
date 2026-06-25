# MacaBrainNet v2

Monkey brain MRI segmentation via a unified multi-class tissue segmentation strategy.

Accurate isolation and classification of brain tissues are critical for cortical surface reconstruction. MacaBrainNet employs a unified multi-class tissue segmentation approach in which brain extraction is implicitly defined by the union of predicted anatomical labels. The model jointly segments cortical gray matter, white matter, cerebrospinal fluid, cerebellum, brainstem, and multiple subcortical structures, generating both a high-fidelity brain mask and anatomically informative labels for downstream analyses.

## Training Data & Strategy

To address the scarcity of annotated macaque MRI data, tissue segmentation labels were developed using an **iterative, pipeline-preserving bootstrapping strategy**. The downstream reconstruction workflow (bias-field correction, volumetric registration, surface initialization, and surface refinement) was kept fixed while only the segmentation component was iteratively improved. Initial labels were obtained by combining deep learning-based cortical tissue segmentation with atlas-informed subcortical labeling after volumetric registration. Cases with suboptimal masks or tissue labels were manually corrected and reprocessed, yielding a curated set of anatomically consistent, surface-informed labels for model training.

Using this strategy, we assembled a **training set of 2,157 macaque MRI scans from 39 acquisition centers**.

## Architecture

The segmentation model is based on **SwinUNETR-B**, a hybrid architecture combining Swin Transformer encoders with CNN decoders, initialized with self-supervised pretraining and fine-tuned for 18-class tissue segmentation (19 including background). Inference uses sliding-window softmax averaging across 5-fold cross-validation ensembles for robust prediction.

| Component | Detail |
|---|---|
| Architecture | SwinUNETR v2, `feature_size=48`, `patch_size=(2,2,2)` |
| Encoder | 4 stages: depths=[2,2,2,2], heads=[3,6,12,24] |
| Decoder | Deep supervision: 5 output heads |
| Pretraining | LocalGlobal self-supervised (17,000 steps) |
| Skull stripping | DiceFocalLoss, 0.5 mm isotropic, patch 96³ |
| Tissue segmentation | Weighted DiceFocalLoss, 0.4 mm isotropic, patch 96³, 19 classes |
| Ensemble | Softmax averaging across 5-fold cross-validation |
| Training data | 2,157 macaque MRI scans from 39 centers |

## Example Results

MacaBrainNet supports multi-modal inputs (T1w, T2w, FLAIR) with a unified model.

### T1w

![Tissue T1w](src/tissue_overlay_T1w.png)
![Mask T1w](src/brain_mask_overlay_T1w.png)

### T2w

![Tissue T2w](src/tissue_overlay_T2w.png)
![Mask T2w](src/brain_mask_overlay_T2w.png)

### FLAIR

![Tissue FLAIR](src/tissue_overlay_FLAIR.png)
![Mask FLAIR](src/brain_mask_overlay_FLAIR.png)

### Label Legend

![Legend](src/tissue_legend.png)

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended, CPU inference supported but slow)
- PyTorch 2.x
- MONAI (for SwinUNETR blocks only)
- nibabel, numpy, scipy

## Installation

```bash
# Clone the repository
git clone https://github.com/yahuiwei123/MacaBrainNet.git
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

The model outputs 19 contiguous classes (0 = background, 1–18 = tissue labels). Labels are derived from FreeSurfer-style anatomical IDs after hemisphere collapsing (left/right merged) and remapped to contiguous 0..18.

### Model Output → FreeSurfer ID Mapping

| Model ID | FS ID | Structure |
|---|---|---|
| 0 | — | Background |
| 1 | 2 | Cerebral White Matter (Left) |
| 2 | 3 | Cerebral Cortex (Left) |
| 3 | 4 | Lateral Ventricle (Left) |
| 4 | 7 | Cerebellum White Matter (Left) |
| 5 | 8 | Cerebellum Cortex (Left) |
| 6 | 10 | Thalamus Proper (Left) |
| 7 | 11 | Caudate (Left) |
| 8 | 12 | Putamen (Left) |
| 9 | 13 | Pallidum (Left) |
| 10 | 16 | Brain Stem |
| 11 | 17 | Hippocampus (Left) |
| 12 | 18 | Amygdala (Left) |
| 13 | 24 | CSF |
| 14 | 26 | Accumbens Area (Left) |
| 15 | 27 | Substantia Nigra (Left) |
| 16 | 28 | Ventral Diencephalon (Left) |
| 17 | 138 | Claustrum (Left) |
| 18 | 140 | Cornea |

> **Hemisphere collapse**: Right-hemisphere labels (41→2, 42→3, 43→4, 46→7, 47→8, 49→10, 50→11, 51→12, 52→13, 53→17, 54→18, 58→26, 59→27, 60→28, 139→138) are merged into their left-hemisphere equivalents before contiguous remapping. Midline structures (Brain Stem, CSF, Cornea) need no collapse.

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

## 5-Fold Cross-Validation Results

Per-class validation Dice scores (mean ± std across 5 folds) from the tissue segmentation model trained with 5-fold cross-validation on 2,157 scans.

| Structure | Dice (mean ± std) | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|---|
| Brain Stem | 0.9228 ± 0.0019 | 0.9232 | 0.9203 | 0.9211 | 0.9254 | 0.9239 |
| Putamen | 0.9043 ± 0.0013 | 0.9054 | 0.9029 | 0.9042 | 0.9062 | 0.9031 |
| Thalamus Proper | 0.9032 ± 0.0011 | 0.9035 | 0.9011 | 0.9043 | 0.9039 | 0.9030 |
| Cerebral WM | 0.8960 ± 0.0019 | 0.8988 | 0.8947 | 0.8962 | 0.8970 | 0.8933 |
| Amygdala | 0.8891 ± 0.0013 | 0.8887 | 0.8871 | 0.8887 | 0.8901 | 0.8910 |
| Cerebral Cortex | 0.8870 ± 0.0023 | 0.8913 | 0.8869 | 0.8868 | 0.8847 | 0.8852 |
| Pallidum | 0.8722 ± 0.0025 | 0.8727 | 0.8690 | 0.8719 | 0.8766 | 0.8709 |
| Caudate | 0.8682 ± 0.0037 | 0.8686 | 0.8628 | 0.8685 | 0.8744 | 0.8670 |
| Hippocampus | 0.8570 ± 0.0037 | 0.8595 | 0.8522 | 0.8547 | 0.8627 | 0.8560 |
| Substantia Nigra | 0.8538 ± 0.0011 | 0.8544 | 0.8523 | 0.8541 | 0.8554 | 0.8527 |
| Accumbens Area | 0.8497 ± 0.0020 | 0.8472 | 0.8488 | 0.8491 | 0.8532 | 0.8501 |
| Ventral Diencephalon | 0.8312 ± 0.0029 | 0.8306 | 0.8275 | 0.8318 | 0.8364 | 0.8299 |
| Cornea | 0.8283 ± 0.0042 | 0.8302 | 0.8207 | 0.8291 | 0.8334 | 0.8281 |
| Cerebellum Cortex | 0.8279 ± 0.0037 | 0.8283 | 0.8249 | 0.8257 | 0.8350 | 0.8254 |
| Claustrum | 0.7463 ± 0.0027 | 0.7490 | 0.7421 | 0.7496 | 0.7451 | 0.7458 |
| CSF | 0.7418 ± 0.0029 | 0.7459 | 0.7411 | 0.7425 | 0.7425 | 0.7368 |
| Cerebellum WM | 0.7340 ± 0.0071 | 0.7330 | 0.7404 | 0.7293 | 0.7433 | 0.7241 |
| Lateral Ventricle | 0.7172 ± 0.0025 | 0.7141 | 0.7189 | 0.7180 | 0.7206 | 0.7145 |
| **Overall** | **0.8404 ± 0.0020** | **0.8413** | **0.8380** | **0.8401** | **0.8437** | **0.8389** |

> Overall Dice includes background class. Sorted by mean Dice descending (excluding background). Consistent performance across folds (std ≤ 0.007) demonstrates stable training.

## OOD Generalization Evaluation

We evaluated the 5-fold ensemble model on an out-of-distribution (OOD) test set of **32 scans from 4 held-out OpenNeuro datasets** that were completely excluded from training:

| OOD Site | Samples | Description |
|---|---|---|
| ds001875 | 9 | Macaque anatomical MRI |
| ds003989 | 13 | Multi-run macaque MRI (3 subjects) |
| ds004620 | 8 | Macaque neuroimaging study |
| ds005521 | 2 | Macaque MRI dataset |

All OOD scans were preprocessed identically (RAS reorientation, 0.4 mm isotropic resampling, brain extraction) and segmented with the 5-fold softmax-averaging ensemble.

### Results

![OOD Results](src/ood_results_figure.png)

**Overall performance: mean Dice = 0.862, mean HD95 = 0.59 mm.**

| Structure | Dice | HD95 (mm) |
|---|---|---|
| Hippocampus | 0.932 | 0.58 |
| Cerebral Cortex | 0.929 | 0.40 |
| Cerebral WM | 0.922 | 0.41 |
| Putamen | 0.917 | 0.52 |
| Thalamus Proper | 0.915 | 0.57 |
| Accumbens Area | 0.906 | 0.55 |
| Pallidum | 0.889 | 0.54 |
| Caudate | 0.886 | 0.56 |
| Amygdala | 0.879 | 0.62 |
| Substantia Nigra | 0.870 | 0.48 |
| Brain Stem | 0.863 | 0.51 |
| CSF | 0.849 | 0.54 |
| Cornea | 0.845 | 0.54 |
| Cerebellum Cortex | 0.822 | 0.80 |
| Ventral Diencephalon | 0.792 | 0.62 |
| Claustrum | 0.785 | 0.50 |
| Cerebellum WM | 0.761 | 1.25 |
| Lateral Ventricle | 0.755 | 0.60 |

Cortical gray/white matter and large subcortical structures achieve Dice > 0.90. Performance is lower for small or thin structures such as Lateral Ventricle, Cerebellum WM, Claustrum, and Ventral Diencephalon, consistent with their anatomical complexity and partial volume effects.

> **Evaluation script**: `evaluate_ood.py` — computes per-class Dice and Hausdorff95 for any JSON-specified dataset using the 5-fold ensemble. Results saved to `ood_results.json`.

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
├── download_from_hf.py        # Download models from HuggingFace
├── src/
│   ├── example/               # Example MRI data (T1w, T2w, FLAIR)
│   ├── run_example.sh         # Example pipeline run
│   ├── make_overlay.py        # Generate overlay visualizations
│   ├── tissue_overlay_*.png   # Example: tissue seg overlays
│   ├── brain_mask_overlay_*.png  # Example: brain mask overlays
│   └── tissue_legend.png      # Tissue class color legend
└── README.md
```

Model checkpoints are stored on HuggingFace Hub (yhwei/MacaBrainNet) and downloaded via `download_from_hf.py`.

## Citation

If you use this toolkit or the pretrained models, please cite:

> Wei, Y. et al. MacaSurfer: unified surface-volume mapping of the macaque brain across the lifespan. 2026.06.14.732101 Preprint at https://doi.org/10.64898/2026.06.14.732101 (2026).
