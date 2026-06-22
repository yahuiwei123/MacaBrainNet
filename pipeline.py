#!/usr/bin/env python3
"""
End-to-end MRI pipeline:
  1. Skull stripping (ensemble, 5-fold)
  2. Brain bbox computation → crop original image +16-voxel padding
  3. Tissue segmentation (ensemble, 5-fold) on cropped image
  4. Remap tissue segmentation back to original image space

Usage:
  python pipeline.py --img T1w.nii.gz --out-dir ./pipeline_output

Model checkpoints are auto-discovered under:
  swinunetr_models/skull_stripping/fold_*/
  swinunetr_models/tissue_segmentation/fold_*/
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict_ensemble import predict_ensemble


def compute_brain_bbox(mask_data, padding=16):
    """
    Compute brain bounding box from binary mask, expanded by `padding` voxels.

    Args:
        mask_data: 3D binary array (0=bg, 1=brain)
        padding: voxels to expand outward on each side

    Returns:
        (start, end) where start=(d0,h0,w0), end=(d1,h1,w1) in voxel coords
    """
    coords = np.argwhere(mask_data > 0)
    if len(coords) == 0:
        raise ValueError("No brain voxels found in mask")

    d_min, h_min, w_min = coords.min(axis=0)
    d_max, h_max, w_max = coords.max(axis=0)

    shape = np.array(mask_data.shape)
    start = np.maximum(0, [d_min - padding, h_min - padding, w_min - padding])
    end = np.minimum(shape, [d_max + padding + 1, h_max + padding + 1, w_max + padding + 1])

    return tuple(start), tuple(end)


def crop_and_mask(img_path, mask_path, padding, out_path):
    """
    Crop original image and brain mask to the expanded brain bbox,
    apply brain mask, and save.

    Args:
        img_path: path to original T1w NIfTI
        mask_path: path to skull-strip mask (binary, same space as img)
        padding: voxels to pad outward from bbox
        out_path: path to save cropped + masked image

    Returns:
        (out_path, start, end) where start/end are the crop slice tuples in original voxel coords
    """
    img = nib.load(img_path)
    mask = nib.load(mask_path)

    img_data = np.asarray(img.dataobj, dtype=np.float32)
    mask_data = np.asarray(mask.dataobj, dtype=np.int16)

    if img_data.shape != mask_data.shape:
        raise ValueError(
            f"Shape mismatch: img {img_data.shape} vs mask {mask_data.shape}. "
            "The skull-strip mask should be in the same space as the input image."
        )

    # Compute expanded bbox
    start, end = compute_brain_bbox(mask_data, padding=padding)
    d0, h0, w0 = start
    d1, h1, w1 = end

    print(f"Brain bbox: ({d0},{h0},{w0}) → ({d1},{h1},{w1})")
    print(f"  Original shape: {img_data.shape}")
    print(f"  Cropped shape:  ({d1 - d0}, {h1 - h0}, {w1 - w0})")

    # Crop image and mask
    cropped_img = img_data[d0:d1, h0:h1, w0:w1].copy()
    cropped_mask = mask_data[d0:d1, h0:h1, w0:w1].copy()

    # Apply brain mask: set non-brain voxels to image minimum
    cropped_img[cropped_mask == 0] = cropped_img[cropped_mask > 0].min()

    # Update affine: voxel (0,0,0) in cropped image corresponds to voxel (d0,h0,w0) in original
    cropped_affine = img.affine.copy()
    cropped_affine[:3, 3] = nib.affines.apply_affine(img.affine, [d0, h0, w0])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(cropped_img, cropped_affine), out_path)
    print(f"Saved cropped+masked image: {out_path}")
    return out_path, start, end


def remap_to_original_space(cropped_seg_path, original_img_path, crop_start, out_path):
    """
    Map a segmentation from cropped space back to full original image space.

    The cropped region is placed at the correct location in a zero-filled
    volume matching the original image dimensions.

    Args:
        cropped_seg_path: path to tissue seg in cropped space
        original_img_path: path to original T1w (for shape + affine reference)
        crop_start: (d0, h0, w0) crop start voxel coordinates in original space
        out_path: path to save remapped segmentation
    """
    cropped_seg = nib.load(cropped_seg_path)
    orig_img = nib.load(original_img_path)

    cropped_data = np.asarray(cropped_seg.dataobj, dtype=np.uint8)
    orig_shape = orig_img.shape

    d0, h0, w0 = crop_start
    cd, ch, cw = cropped_data.shape

    # Place cropped segmentation back into full-size zero volume
    full_seg = np.zeros(orig_shape, dtype=np.uint8)
    full_seg[d0:d0 + cd, h0:h0 + ch, w0:w0 + cw] = cropped_data

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(full_seg, orig_img.affine), out_path)
    print(f"Saved full-space tissue seg: {out_path}")
    return out_path


def run_pipeline(img_path, out_dir,
                 skull_ckpt_dir=None,
                 tissue_ckpt_dir=None,
                 skull_spacing=(0.5, 0.5, 0.5),
                 tissue_spacing=(0.4, 0.4, 0.4),
                 patch_size=(96, 96, 96),
                 overlap=0.60,
                 padding=16,
                 device=None):
    """
    Run the full skull-strip → crop → tissue-seg pipeline.

    Args:
        img_path: input T1w NIfTI
        out_dir: output directory for all results
        skull_ckpt_dir: dir with fold_*/ for skull stripping model
        tissue_ckpt_dir: dir with fold_*/ for tissue segmentation model
        skull_spacing: target spacing for skull stripping (default 0.5mm)
        tissue_spacing: target spacing for tissue segmentation (default 0.4mm)
        patch_size: patch size for sliding window
        overlap: sliding window overlap ratio
        padding: voxels to pad outward from brain bbox
        device: torch device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(img_path))[0]
    if basename.endswith(".nii"):
        basename = basename[:-4]

    # ── Default checkpoint directories ──
    if skull_ckpt_dir is None:
        skull_ckpt_dir = os.path.join(project_root, "swinunetr_models", "skull_stripping")
    if tissue_ckpt_dir is None:
        tissue_ckpt_dir = os.path.join(project_root, "swinunetr_models", "tissue_segmentation")

    # =====================================================================
    # Step 1: Skull Stripping (ensemble)
    # =====================================================================
    print("=" * 60)
    print("STEP 1: Skull Stripping (ensemble)")
    print("=" * 60)

    skull_mask_path = os.path.join(out_dir, f"{basename}_brain_mask.nii.gz")

    predict_ensemble(
        img_path=img_path,
        out_path=skull_mask_path,
        ckpt_dir=skull_ckpt_dir,
        target_spacing=skull_spacing,
        patch_size=patch_size,
        overlap=overlap,
        num_classes=2,
        do_clean=True,
        device=device,
    )

    # =====================================================================
    # Step 2: Crop & Brain Mask
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Brain BBox Crop + Mask")
    print("=" * 60)

    cropped_img_path = os.path.join(out_dir, f"{basename}_cropped.nii.gz")
    _, crop_start, crop_end = crop_and_mask(
        img_path, skull_mask_path, padding, cropped_img_path)

    # =====================================================================
    # Step 3: Tissue Segmentation (ensemble) on cropped image
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Tissue Segmentation (ensemble)")
    print("=" * 60)

    tissue_cropped_path = os.path.join(out_dir, f"{basename}_tissue_seg_cropped.nii.gz")

    predict_ensemble(
        img_path=cropped_img_path,
        out_path=tissue_cropped_path,
        ckpt_dir=tissue_ckpt_dir,
        target_spacing=tissue_spacing,
        patch_size=patch_size,
        overlap=overlap,
        num_classes=19,
        do_clean=True,
        device=device,
    )

    # =====================================================================
    # Step 4: Remap tissue seg back to original image space
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Remap to Original Space")
    print("=" * 60)

    tissue_full_path = os.path.join(out_dir, f"{basename}_tissue_seg.nii.gz")

    remap_to_original_space(
        tissue_cropped_path, img_path, crop_start, tissue_full_path)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Brain mask:        {skull_mask_path}")
    print(f"  Cropped img:       {cropped_img_path}")
    print(f"  Tissue seg (crop): {tissue_cropped_path}")
    print(f"  Tissue seg (full): {tissue_full_path}")

    return skull_mask_path, cropped_img_path, tissue_full_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Skull Stripping → BBox Crop → Tissue Segmentation Pipeline")
    parser.add_argument("--img", type=str, required=True,
                        help="Input T1w NIfTI image")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for all results")
    parser.add_argument("--skull-ckpt-dir", type=str, default=None,
                        help="Dir containing fold_*/ for skull stripping "
                        "(default: swinunetr_models/skull_stripping)")
    parser.add_argument("--tissue-ckpt-dir", type=str, default=None,
                        help="Dir containing fold_*/ for tissue segmentation "
                        "(default: swinunetr_models/tissue_segmentation)")
    parser.add_argument("--skull-spacing", type=float, nargs=3,
                        default=[0.5, 0.5, 0.5])
    parser.add_argument("--tissue-spacing", type=float, nargs=3,
                        default=[0.4, 0.4, 0.4])
    parser.add_argument("--patch-size", type=int, nargs=3,
                        default=[96, 96, 96])
    parser.add_argument("--overlap", type=float, default=0.60)
    parser.add_argument("--padding", type=int, default=16,
                        help="Voxels to pad outward from brain bbox (default: 16)")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    run_pipeline(
        img_path=args.img,
        out_dir=args.out_dir,
        skull_ckpt_dir=args.skull_ckpt_dir,
        tissue_ckpt_dir=args.tissue_ckpt_dir,
        skull_spacing=tuple(args.skull_spacing),
        tissue_spacing=tuple(args.tissue_spacing),
        patch_size=tuple(args.patch_size),
        overlap=args.overlap,
        padding=args.padding,
        device=device,
    )
