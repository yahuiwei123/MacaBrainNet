#!/usr/bin/env python3
"""
Ensemble prediction: runs inference across all cross-validation fold models,
averages softmax probabilities, then argmax for the final segmentation.
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nets.swinunetr import SwinUNETR
from trainer import sliding_window_inference, _safe_torch_load
from data.transforms import (
    load_nifti, ensure_channel_first, reorient_to_ras,
    resample_to_spacing, clip_and_normalize, spatial_pad,
)
from predict import (
    load_model, preprocess, postprocess, clean_mask,
)


def find_fold_ckpts(ckpt_dir: str, pattern: str = "best_3d_swinunetr_model.pth"):
    """Find all fold checkpoints under ckpt_dir, sorted by fold number."""
    ckpts = sorted(
        glob.glob(os.path.join(ckpt_dir, "fold_*", pattern)),
        key=lambda p: int(os.path.basename(os.path.dirname(p)).replace("fold_", "")),
    )
    if not ckpts:
        # try flat pattern: ckpt_dir/*.pth
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return ckpts


def predict_ensemble(img_path: str, out_path: str, ckpt_dir: str,
                     target_spacing: tuple = (0.5, 0.5, 0.5),
                     patch_size: tuple = (96, 96, 96),
                     overlap: float = 0.60,
                     num_classes: int = 2,
                     do_clean: bool = True,
                     device: torch.device = None):
    """Run ensemble prediction across all fold models."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Preprocess once (all folds share the same input) ──
    print(f"Preprocessing: {img_path}")
    tensor, orig_affine, orig_shape, pre_pad_shape, ras_shape, orig_axcodes = \
        preprocess(img_path, target_spacing, patch_size)
    tensor = tensor.to(device)
    print(f"  Input shape: {tuple(tensor.shape)}")

    # ── Find fold checkpoints ──
    ckpt_paths = find_fold_ckpts(ckpt_dir)
    print(f"\nFound {len(ckpt_paths)} fold checkpoints:")
    for p in ckpt_paths:
        print(f"  {p}")

    # ── Ensemble inference ──
    prob_sum = None

    for idx, ckpt_path in enumerate(ckpt_paths, 1):
        fold_name = os.path.basename(os.path.dirname(ckpt_path))
        print(f"\n[{idx}/{len(ckpt_paths)}] Loading {fold_name}: {ckpt_path}")
        model = load_model(ckpt_path, device, num_classes=num_classes,
                           patch_size=patch_size)

        print(f"  Running inference...")
        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=tensor,
                roi_size=patch_size,
                network=model,
                overlap=overlap,
            )

        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        # softmax → probability
        prob = F.softmax(logits, dim=1)  # [1, C, D, H, W]

        if prob_sum is None:
            prob_sum = prob
        else:
            prob_sum += prob

        # Free model memory
        del model, logits, prob
        torch.cuda.empty_cache()

    # ── Average & argmax ──
    prob_mean = prob_sum / len(ckpt_paths)
    pred = torch.argmax(prob_mean, dim=1)  # [1, D, H, W]
    pred_np = pred[0].cpu().numpy().astype(np.uint8)

    print(f"\nEnsemble complete — averaged {len(ckpt_paths)} models.")

    # ── Postprocess ──
    pred_np, save_affine = postprocess(
        pred_np, orig_affine, orig_shape,
        pre_pad_shape, ras_shape, orig_axcodes)

    if do_clean:
        print("Cleaning mask (largest CC + fill holes)...")
        pred_np = clean_mask(pred_np, num_classes)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(pred_np, save_affine), out_path)
    print(f"Saved: {out_path}")
    return pred_np, save_affine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble prediction — averages softmax across all fold models")
    parser.add_argument("--img", type=str, required=True, help="Input NIfTI image")
    parser.add_argument("--out", type=str, required=True, help="Output NIfTI mask")
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Directory containing fold_*/best_3d_swinunetr_model.pth")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--spacing", type=float, nargs=3, default=[0.5, 0.5, 0.5])
    parser.add_argument("--overlap", type=float, default=0.60)
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of output classes")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-clean", action="store_true",
                        help="Skip mask cleaning")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    predict_ensemble(
        img_path=args.img,
        out_path=args.out,
        ckpt_dir=args.ckpt_dir,
        target_spacing=tuple(args.spacing),
        patch_size=tuple(args.patch_size),
        overlap=args.overlap,
        num_classes=args.num_classes,
        do_clean=not args.no_clean,
        device=device,
    )
