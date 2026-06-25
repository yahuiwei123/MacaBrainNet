#!/usr/bin/env python3
"""
Evaluate 5-fold ensemble on OOD tissue segmentation dataset.
Reports per-class Dice and Hausdorff95 distance.

Usage:
    python evaluate_ood.py --ckpt-dir swinunetr_models/tissue_segmentation --device cuda
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nets.swinunetr import SwinUNETR
from trainer import sliding_window_inference, _safe_torch_load
from data.transforms import (
    load_nifti, ensure_channel_first, reorient_to_ras,
    resample_to_spacing, clip_and_normalize, spatial_pad,
)
from predict_ensemble import find_fold_ckpts

# Tissue segmentation: contiguous label id → name
LABEL_NAMES = [
    "Background",                   # 0
    "Cerebral WM (L)",              # 1  → FS 2
    "Cerebral Cortex (L)",          # 2  → FS 3
    "Lateral Ventricle (L)",        # 3  → FS 4
    "Cerebellum WM (L)",            # 4  → FS 7
    "Cerebellum Cortex (L)",        # 5  → FS 8
    "Thalamus Proper (L)",          # 6  → FS 10
    "Caudate (L)",                  # 7  → FS 11
    "Putamen (L)",                  # 8  → FS 12
    "Pallidum (L)",                 # 9  → FS 13
    "Brain Stem",                   # 10 → FS 16
    "Hippocampus (L)",              # 11 → FS 17
    "Amygdala (L)",                 # 12 → FS 18
    "CSF",                          # 13 → FS 24
    "Accumbens Area (L)",           # 14 → FS 26
    "Substantia Nigra (L)",         # 15 → FS 27
    "Ventral Diencephalon (L)",     # 16 → FS 28
    "Claustrum (L)",                # 17 → FS 138
    "Cornea",                       # 18 → FS 140
]

NUM_CLASSES = 19
TARGET_SPACING = (0.4, 0.4, 0.4)
PATCH_SIZE = (96, 96, 96)
OVERLAP = 0.60


def load_model(ckpt_path, device):
    model = SwinUNETR(
        img_size=PATCH_SIZE,
        in_channels=1,
        out_channels=NUM_CLASSES,
        patch_size=(2, 2, 2),
        feature_size=48,
        use_v2=True,
    ).to(device)

    ckpt = _safe_torch_load(ckpt_path, map_location=device, trust_source=True)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt

    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def preprocess_volume(img_path, label_path, data_root):
    """
    Preprocess image and label through the same spatial pipeline.
    Data is preprocessed (RAS, 0.4mm, labels 0..18), so reorient/resample
    are effectively identity — but we run them for correctness.

    Returns:
        img_tensor: [1, 1, Dp, Hp, Wp] padded
        label_padded: [1, Dp, Hp, Wp] padded
        pre_pad_shape: spatial shape before padding
    """
    full_img = os.path.join(data_root, img_path) if data_root else img_path
    full_lbl = os.path.join(data_root, label_path) if data_root else label_path

    # ── Image ──
    img, img_aff = load_nifti(full_img)
    img = ensure_channel_first(img)
    img, ras_aff = reorient_to_ras(img, img_aff)
    img, _ = resample_to_spacing(img, ras_aff, TARGET_SPACING, order=3)
    pre_pad_shape = img.shape[-3:]
    img = np.ascontiguousarray(img)
    img = clip_and_normalize(img)
    img = spatial_pad(img, PATCH_SIZE, mode='minimum')
    img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # [1,1,D,H,W]

    # ── Label ──
    lbl, lbl_aff = load_nifti(full_lbl)
    lbl = ensure_channel_first(lbl)
    lbl, _ = reorient_to_ras(lbl, lbl_aff)
    lbl, _ = resample_to_spacing(lbl, ras_aff, TARGET_SPACING, order=0, is_label=True)
    lbl = np.ascontiguousarray(lbl)
    lbl = spatial_pad(lbl, PATCH_SIZE, mode='constant', constant_values=0)
    lbl_tensor = torch.from_numpy(lbl).long()  # [1, Dp, Hp, Wp]

    return img_tensor, lbl_tensor, pre_pad_shape


def unpad_volume(vol, pre_pad_shape):
    """Crop padded volume back to pre_pad_shape."""
    D, H, W = vol.shape[-3:]
    pD, pH, pW = pre_pad_shape
    d0 = (D - pD) // 2
    h0 = (H - pH) // 2
    w0 = (W - pW) // 2
    return vol[..., d0:d0+pD, h0:h0+pH, w0:w0+pW]


def per_class_dice(pred, label, num_classes):
    """
    pred, label: [D, H, W] integer arrays.
    Returns dice [num_classes], present [num_classes].
    """
    dice = np.zeros(num_classes, dtype=np.float64)
    present = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        lab_c = (label == c)
        if not lab_c.any():
            continue
        present[c] = 1.0
        pred_c = (pred == c)
        inter = (pred_c & lab_c).sum()
        denom = pred_c.sum() + lab_c.sum()
        dice[c] = (2.0 * inter + 1e-5) / (denom + 1e-5)

    return dice, present


def per_class_hausdorff95(pred, label, num_classes, spacing):
    """
    Surface-based Hausdorff distance at 95th percentile.
    Uses scipy.ndimage.distance_transform_edt for efficiency.

    Returns hd95 [num_classes], NaN where class not present.
    """
    hd95 = np.full(num_classes, np.nan)
    spacing = np.array(spacing, dtype=np.float64)

    for c in range(1, num_classes):  # skip background
        pred_c = (pred == c)
        label_c = (label == c)

        if not label_c.any():
            continue
        if not pred_c.any():
            hd95[c] = np.inf
            continue

        # Extract surfaces via erosion
        pred_surf = pred_c ^ binary_erosion(pred_c)
        lbl_surf = label_c ^ binary_erosion(label_c)

        if not pred_surf.any() or not lbl_surf.any():
            # Fallback: use all voxels if surface extraction yields empty set
            pred_surf = pred_c
            lbl_surf = label_c

        # Distance from pred surface → label surface
        d_to_lbl = distance_transform_edt(~lbl_surf, sampling=spacing)
        d1 = d_to_lbl[pred_surf]

        # Distance from label surface → pred surface
        d_to_pred = distance_transform_edt(~pred_surf, sampling=spacing)
        d2 = d_to_pred[lbl_surf]

        hd95[c] = np.percentile(np.concatenate([d1, d2]), 95)

    return hd95


def run_ensemble(tensor, ckpt_paths, device):
    """Run 5-fold ensemble inference. Returns logits [1, C, D, H, W]."""
    prob_sum = None

    for idx, ckpt_path in enumerate(ckpt_paths, 1):
        fold_name = os.path.basename(os.path.dirname(ckpt_path))
        print(f"    [{idx}/{len(ckpt_paths)}] {fold_name}...", end=" ", flush=True)

        model = load_model(ckpt_path, device)

        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=tensor.to(device),
                roi_size=PATCH_SIZE,
                network=model,
                overlap=OVERLAP,
            )

        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        prob = F.softmax(logits, dim=1).cpu()

        if prob_sum is None:
            prob_sum = prob
        else:
            prob_sum += prob

        del model, logits, prob
        torch.cuda.empty_cache()
        print("done")

    prob_mean = prob_sum / len(ckpt_paths)
    pred = torch.argmax(prob_mean, dim=1)  # [1, D, H, W]
    return pred


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 5-fold ensemble on OOD tissue segmentation dataset")
    parser.add_argument("--ckpt-dir", type=str,
                        default="swinunetr_models/tissue_segmentation",
                        help="Directory containing fold_*/best_3d_swinunetr_model.pth")
    parser.add_argument("--json", type=str,
                        default="/home/weiyahui/projects/monkey/dataset/tissue_segmentation/ood_test.json",
                        help="Path to OOD test JSON")
    parser.add_argument("--data-root", type=str,
                        default="/home/weiyahui/projects/monkey/dataset/tissue_segmentation",
                        help="Root directory for relative paths in JSON")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save detailed results JSON")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load OOD sample list
    with open(args.json, 'r') as f:
        data = json.load(f)
    samples = data.get("validation", data.get("training", []))
    print(f"OOD samples: {len(samples)}")

    # Find fold checkpoints
    ckpt_paths = find_fold_ckpts(args.ckpt_dir)
    print(f"Fold checkpoints ({len(ckpt_paths)}):")
    for p in ckpt_paths:
        print(f"  {p}")

    # ── Evaluate each sample ──
    all_dice = []        # list of [num_classes]
    all_present = []     # list of [num_classes]
    all_hd95 = []        # list of [num_classes]
    sample_names = []

    for i, sample in enumerate(samples):
        name = sample["image"].replace("site-", "").replace("/anat/", "/")
        sample_names.append(name)
        print(f"\n[{i+1}/{len(samples)}] {name}")

        # Preprocess
        img_tensor, lbl_tensor, pre_pad_shape = preprocess_volume(
            sample["image"], sample["label"], args.data_root)

        print(f"    volume: {img_tensor.shape}, pre_pad: {pre_pad_shape}")

        # Ensemble inference
        pred_padded = run_ensemble(img_tensor, ckpt_paths, device)  # [1, Dp, Hp, Wp]

        # Unpad prediction and label
        pred_np = unpad_volume(pred_padded, pre_pad_shape).squeeze(0).cpu().numpy().astype(np.int64)
        lbl_np = unpad_volume(lbl_tensor, pre_pad_shape).squeeze(0).cpu().numpy().astype(np.int64)

        # Compute metrics
        dice, present = per_class_dice(pred_np, lbl_np, NUM_CLASSES)
        hd95 = per_class_hausdorff95(pred_np, lbl_np, NUM_CLASSES, TARGET_SPACING)

        all_dice.append(dice)
        all_present.append(present)
        all_hd95.append(hd95)

        # Per-sample summary
        fg_dice = dice[1:][present[1:] > 0]
        mean_dice = fg_dice.mean() if len(fg_dice) > 0 else 0.0
        print(f"    mean Dice (fg): {mean_dice:.4f}")

    # ── Aggregate results ──
    dice_stack = np.stack(all_dice, axis=0)     # [N, num_classes]
    present_stack = np.stack(all_present, axis=0)
    hd95_stack = np.stack(all_hd95, axis=0)

    print("\n" + "=" * 80)
    print("OOD Evaluation Results — 5-fold Ensemble")
    print("=" * 80)
    print(f"{'Class':<20s} {'Dice':>8s}  {'HD95(mm)':>10s}  {'#Present':>8s}")
    print("-" * 54)

    for c in range(NUM_CLASSES):
        n_present = int(present_stack[:, c].sum())
        if n_present == 0:
            print(f"{LABEL_NAMES[c]:<20s} {'N/A':>8s}  {'N/A':>10s}  {n_present:>8d}")
            continue

        d_vals = dice_stack[:, c]
        d_mean = d_vals[d_vals > 0].mean() if (d_vals > 0).any() else 0.0

        h_vals = hd95_stack[:, c]
        h_valid = h_vals[~np.isnan(h_vals) & ~np.isinf(h_vals)]
        h_mean = h_valid.mean() if len(h_valid) > 0 else float('nan')

        print(f"{LABEL_NAMES[c]:<20s} {d_mean:>8.4f}  {h_mean:>10.2f}  {n_present:>8d}")

    # Overall (non-background, excluding bg)
    fg_mask = (present_stack[:, 1:].sum(axis=0) > 0)
    fg_dice_all = dice_stack[:, 1:][:, fg_mask]
    fg_hd95_all = hd95_stack[:, 1:][:, fg_mask]
    fg_hd95_valid = fg_hd95_all[~np.isnan(fg_hd95_all) & ~np.isinf(fg_hd95_all)]

    overall_dice = fg_dice_all[fg_dice_all > 0].mean() if (fg_dice_all > 0).any() else 0.0
    overall_hd95 = fg_hd95_valid.mean() if len(fg_hd95_valid) > 0 else float('nan')

    print("-" * 54)
    print(f"{'Overall (fg mean)':<20s} {overall_dice:>8.4f}  {overall_hd95:>10.2f}")
    print("=" * 80)

    # ── Save detailed JSON ──
    if args.output:
        results = {
            "num_samples": len(samples),
            "num_classes": NUM_CLASSES,
            "label_names": LABEL_NAMES,
            "overall_dice_mean": float(overall_dice),
            "overall_hd95_mean": float(overall_hd95) if not np.isnan(overall_hd95) else None,
            "per_class": {},
            "per_sample": [],
        }

        for c in range(NUM_CLASSES):
            n_p = int(present_stack[:, c].sum())
            d_vals = dice_stack[:, c]
            d_mean = float(d_vals[d_vals > 0].mean()) if (d_vals > 0).any() else None
            d_std = float(d_vals[d_vals > 0].std()) if (d_vals > 0).any() else None
            h_vals = hd95_stack[:, c]
            h_valid = h_vals[~np.isnan(h_vals) & ~np.isinf(h_vals)]
            h_mean = float(h_valid.mean()) if len(h_valid) > 0 else None
            h_std = float(h_valid.std()) if len(h_valid) > 0 else None

            results["per_class"][LABEL_NAMES[c]] = {
                "id": c,
                "dice_mean": d_mean,
                "dice_std": d_std,
                "hd95_mean_mm": h_mean,
                "hd95_std_mm": h_std,
                "n_present": n_p,
            }

        for i, sample in enumerate(samples):
            results["per_sample"].append({
                "image": sample["image"],
                "label": sample["label"],
                "dice_per_class": {LABEL_NAMES[c]: float(all_dice[i][c])
                                   for c in range(NUM_CLASSES)},
                "hd95_per_class": {LABEL_NAMES[c]: float(all_hd95[i][c])
                                   if not np.isnan(all_hd95[i][c]) else None
                                   for c in range(NUM_CLASSES)},
            })

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
