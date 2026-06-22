#!/usr/bin/env python3
"""
Single-checkpoint prediction for skull stripping / tissue segmentation.
Supports binary (skull strip) and multi-class (tissue segmentation) models.
No MONAI dependency — uses custom transforms + sliding_window_inference.
"""

import os
import sys
import argparse
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom, label, binary_fill_holes

# Make sure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nets.swinunetr import SwinUNETR
from trainer import sliding_window_inference, _safe_torch_load
from data.transforms import (
    load_nifti, ensure_channel_first, reorient_to_ras,
    resample_to_spacing, clip_and_normalize, spatial_pad,
)


def load_model(ckpt_path: str, device: torch.device, num_classes: int = 2,
               patch_size: tuple = (96, 96, 96)):
    """Build SwinUNETR and load checkpoint weights."""
    model = SwinUNETR(
        img_size=patch_size,
        in_channels=1,
        out_channels=num_classes,
        patch_size=(2, 2, 2),
        feature_size=48,
        use_v2=True,
    ).to(device)

    ckpt = _safe_torch_load(ckpt_path, map_location=device, trust_source=True)

    if isinstance(ckpt, dict):
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt

    # strip DDP prefix
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def preprocess(img_path: str, target_spacing: tuple, patch_size: tuple):
    """
    Preprocess a NIfTI image for inference.
    Returns (tensor, original_affine, original_shape, pre_pad_shape,
             ras_shape, original_axcodes)
    """
    # Load
    data, affine = load_nifti(img_path)
    original_shape = data.shape
    original_affine = affine.copy()
    original_axcodes = nib.orientations.aff2axcodes(original_affine)

    # Channel first
    data = ensure_channel_first(data)  # [1, D, H, W]

    # Reorient to RAS
    data, ras_affine = reorient_to_ras(data, affine)
    ras_shape = data.shape[-3:]  # spatial shape in RAS orientation

    # Resample to target spacing
    data, resampled_affine = resample_to_spacing(data, ras_affine, target_spacing, order=3)
    pre_pad_shape = data.shape[-3:]  # spatial shape before padding

    # Normalize
    data = np.ascontiguousarray(data)
    data = clip_and_normalize(data)

    # Pad to patch_size
    data = spatial_pad(data, patch_size, mode='minimum')

    tensor = torch.from_numpy(data).float().unsqueeze(0)  # [1, 1, D, H, W]
    return tensor, original_affine, original_shape, pre_pad_shape, ras_shape, original_axcodes


def postprocess(pred_np, original_affine, original_shape, pre_pad_shape,
                ras_shape, original_axcodes):
    """
    Inverse preprocessing: unpad → unresample → unreorient.
    Returns (data, affine) in original image space.
    """
    # 1. Unpad: crop back to pre_pad_shape
    pad_before = tuple((p - r) // 2 for p, r in zip(pred_np.shape, pre_pad_shape))
    pred_unpadded = pred_np[
        pad_before[0]:pad_before[0] + pre_pad_shape[0],
        pad_before[1]:pad_before[1] + pre_pad_shape[1],
        pad_before[2]:pad_before[2] + pre_pad_shape[2],
    ]

    # 2. Resample back: pre_pad_shape → ras_shape (invert resample_to_spacing)
    if tuple(pre_pad_shape) != tuple(ras_shape):
        zoom_factors = tuple(r / p for r, p in zip(ras_shape, pre_pad_shape))
        pred_ras = zoom(pred_unpadded.astype(np.float32), zoom_factors, order=0)
    else:
        pred_ras = pred_unpadded

    # 3. Reorient from RAS back to original orientation
    if original_axcodes != ('R', 'A', 'S'):
        ras_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
        orig_ornt = nib.orientations.axcodes2ornt(original_axcodes)
        inv_transform = nib.orientations.ornt_transform(ras_ornt, orig_ornt)
        if not np.allclose(inv_transform, [[0, 1], [1, 1], [2, 1]]):
            pred_original = nib.orientations.apply_orientation(pred_ras, inv_transform)
        else:
            pred_original = pred_ras
    else:
        pred_original = pred_ras

    return pred_original.astype(np.uint8), original_affine


def _largest_connected_component(mask):
    """Keep only the largest connected component in a binary mask."""
    labeled, num_features = label(mask)
    if num_features <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())
    if len(sizes) <= 1:
        return mask
    largest_label = sizes[1:].argmax() + 1  # skip background (label 0)
    return labeled == largest_label


def clean_mask(pred_np, num_classes):
    """
    Post-process a predicted label map:
    - Keep only the largest foreground connected component (removes isolated noise).
    - Fill holes per-class and combine (higher class ID takes priority).
    """
    if num_classes == 2:
        fg = (pred_np == 1)
        if not fg.any():
            return pred_np
        fg = _largest_connected_component(fg)
        fg = binary_fill_holes(fg)
        return fg.astype(np.uint8)

    # ── Multi-class ──────────────────────────────────────────────
    # 1. Largest CC on overall foreground (preserves all internal structures)
    fg_all = (pred_np > 0)
    fg_all = _largest_connected_component(fg_all)
    pred_np = pred_np.copy()
    pred_np[~fg_all] = 0

    # 2. Per-class fill holes, higher class overwrites
    result = np.zeros_like(pred_np)
    for c in range(1, num_classes):
        fg = (pred_np == c)
        if not fg.any():
            continue
        fg = binary_fill_holes(fg)
        result[fg] = c
    return result


def predict_single(img_path: str, out_path: str, ckpt_path: str,
                   target_spacing: tuple = (0.5, 0.5, 0.5),
                   patch_size: tuple = (96, 96, 96),
                   overlap: float = 0.60,
                   num_classes: int = 2,
                   do_clean: bool = True,
                   device: torch.device = None):
    """Run prediction on a single image."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model: {ckpt_path}")
    model = load_model(ckpt_path, device, num_classes=num_classes,
                       patch_size=patch_size)

    # Preprocess
    print(f"Preprocessing: {img_path}")
    tensor, orig_affine, orig_shape, pre_pad_shape, ras_shape, orig_axcodes = \
        preprocess(img_path, target_spacing, patch_size)
    tensor = tensor.to(device)
    print(f"  Input shape: {tuple(tensor.shape)}")

    # Inference
    print("Running inference...")
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=tensor,
            roi_size=patch_size,
            network=model,
            overlap=overlap,
        )

    if isinstance(output, (list, tuple)):
        output = output[0]

    # For binary skull stripping: argmax to get 0/1 mask
    if num_classes == 2:
        pred = torch.argmax(output, dim=1)  # [B, D, H, W]
        pred_np = pred[0].cpu().numpy().astype(np.uint8)
    else:
        # Multi-class: map contig ids back (caller handles mapping)
        pred = torch.argmax(output, dim=1)
        pred_np = pred[0].cpu().numpy().astype(np.uint8)

    # Postprocess: reverse preprocessing to get back to original image space
    pred_np, save_affine = postprocess(
        pred_np, orig_affine, orig_shape,
        pre_pad_shape, ras_shape, orig_axcodes)

    # Mask cleaning: largest connected component + fill holes
    if do_clean:
        print("Cleaning mask (largest CC + fill holes)...")
        pred_np = clean_mask(pred_np, num_classes)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(pred_np, save_affine), out_path)
    print(f"Saved: {out_path}")
    return pred_np, save_affine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Checkpoint Skull Stripping Prediction")
    parser.add_argument("--img", type=str, required=True, help="Input NIfTI image")
    parser.add_argument("--out", type=str, required=True, help="Output NIfTI mask")
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--spacing", type=float, nargs=3, default=[0.5, 0.5, 0.5])
    parser.add_argument("--overlap", type=float, default=0.60)
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of output classes (2 for skull strip, 19 for tissue segmentation)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--no-clean", action="store_true",
                        help="Skip mask cleaning (largest CC + fill holes)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    predict_single(
        img_path=args.img,
        out_path=args.out,
        ckpt_path=args.ckpt,
        target_spacing=tuple(args.spacing),
        patch_size=tuple(args.patch_size),
        overlap=args.overlap,
        num_classes=args.num_classes,
        do_clean=not args.no_clean,
        device=device,
    )
