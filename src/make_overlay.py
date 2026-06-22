#!/usr/bin/env python3
"""Generate overlay images of tissue segmentation on original MRI slices."""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.transforms import load_nifti, ensure_channel_first, reorient_to_ras


# 19-class colormap (class 0=background transparent, 1-18 colored)
CLASS_COLORS = [
    (0.0,   0.0,   0.0),    # 0: background
    (1.0,   0.9,   0.3),    # 1: Cerebral WM
    (0.4,   0.8,   0.4),    # 2: Cerebral Cortex
    (0.2,   0.5,   1.0),    # 3: Lateral Ventricle
    (0.3,   0.7,   0.9),    # 4: Inf-Lat-Vent
    (0.9,   0.7,   0.2),    # 5: Cerebellum WM
    (0.5,   1.0,   0.5),    # 6: Cerebellum Cortex
    (1.0,   0.7,   0.4),    # 7: Thalamus Proper
    (1.0,   0.3,   0.3),    # 8: Caudate
    (1.0,   0.5,   0.5),    # 9: Putamen
    (0.8,   0.4,   0.4),    # 10: Pallidum
    (0.3,   0.8,   0.8),    # 11: Hippocampus
    (1.0,   0.5,   0.8),    # 12: Amygdala
    (0.2,   0.9,   1.0),    # 13: CSF
    (0.9,   0.3,   0.5),    # 14: Accumbens Area
    (0.4,   0.4,   0.4),    # 15: Substantia Nigra
    (0.5,   0.5,   0.9),    # 16: Ventral DC
    (0.7,   0.9,   0.4),    # 17: Choroid Plexus
    (0.6,   0.4,   0.8),    # 18: Claustrum
]

CLASS_NAMES = [
    "Background",
    "Cerebral WM", "Cerebral Cortex", "Lateral Ventricle",
    "Inf-Lat-Vent", "Cerebellum WM", "Cerebellum Cortex",
    "Thalamus Proper", "Caudate", "Putamen", "Pallidum",
    "Hippocampus", "Amygdala", "CSF",
    "Accumbens Area", "Substantia Nigra", "Ventral DC",
    "Choroid Plexus", "Claustrum",
]


def make_overlay(orig_path, mask_path, seg_path, out_dir, modality="T1w"):
    os.makedirs(out_dir, exist_ok=True)

    orig = nib.load(orig_path)
    seg = nib.load(seg_path)

    orig_data = np.asarray(orig.dataobj, dtype=np.float32)
    seg_data = np.asarray(seg.dataobj, dtype=np.uint8)

    # Normalize original to [0,1]
    p_low, p_high = np.percentile(orig_data[orig_data > 0], [0.5, 99.5])
    orig_norm = np.clip(orig_data, p_low, p_high)
    orig_norm = (orig_norm - p_low) / (p_high - p_low + 1e-8)

    D, H, W = orig_data.shape

    cmap = ListedColormap(CLASS_COLORS[:max(seg_data.max() + 1, 2)])

    # Generate slices: axial (z), coronal (y), sagittal (x)
    slices_axial = [int(D * 0.25), D // 2, int(D * 0.75)]
    slices_coronal = [int(H * 0.25), H // 2, int(H * 0.75)]
    slices_sagittal = [int(W * 0.25), W // 2, int(W * 0.75)]

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    plt.subplots_adjust(wspace=0.02, hspace=0.06)

    # Axial row
    for i, s in enumerate(slices_axial):
        ax = axes[0, i]
        ax.imshow(orig_norm[s, :, :].T, cmap="gray", origin="lower", aspect="auto")
        seg_slice = seg_data[s, :, :].T.astype(int)
        masked = np.ma.masked_where(seg_slice == 0, seg_slice)
        ax.imshow(masked, cmap=cmap, vmin=0, vmax=18, origin="lower",
                  aspect="auto", alpha=0.6, interpolation="nearest")
        ax.set_title(f"Axial z={s}/{D}", fontsize=10)
        ax.axis("off")

    # Coronal row
    for i, s in enumerate(slices_coronal):
        ax = axes[1, i]
        ax.imshow(orig_norm[:, s, :].T, cmap="gray", origin="lower", aspect="auto")
        seg_slice = seg_data[:, s, :].T.astype(int)
        masked = np.ma.masked_where(seg_slice == 0, seg_slice)
        ax.imshow(masked, cmap=cmap, vmin=0, vmax=18, origin="lower",
                  aspect="auto", alpha=0.6, interpolation="nearest")
        ax.set_title(f"Coronal y={s}/{H}", fontsize=10)
        ax.axis("off")

    # Sagittal row
    for i, s in enumerate(slices_sagittal):
        ax = axes[2, i]
        ax.imshow(orig_norm[:, :, s].T, cmap="gray", origin="lower", aspect="auto")
        seg_slice = seg_data[:, :, s].T.astype(int)
        masked = np.ma.masked_where(seg_slice == 0, seg_slice)
        ax.imshow(masked, cmap=cmap, vmin=0, vmax=18, origin="lower",
                  aspect="auto", alpha=0.6, interpolation="nearest")
        ax.set_title(f"Sagittal x={s}/{W}", fontsize=10)
        ax.axis("off")

    plt.suptitle(f"MacaBrainNet Tissue Segmentation ({modality})", fontsize=14, y=0.98)
    overlay_path = os.path.join(out_dir, f"tissue_overlay_{modality}.png")
    plt.savefig(overlay_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {overlay_path}")

    # Legend
    fig_legend, ax = plt.subplots(1, 1, figsize=(16, 2))
    present_classes = sorted(set(np.unique(seg_data)))
    for c in present_classes:
        if c == 0:
            continue
        color = CLASS_COLORS[c]
        ax.add_patch(plt.Rectangle((c * 0.9, 0), 0.8, 1, facecolor=color,
                                    edgecolor="gray", linewidth=0.5))
        ax.text(c * 0.9 + 0.4, 0.5, CLASS_NAMES[c], ha="center", va="center",
                fontsize=6, rotation=90 if len(CLASS_NAMES[c]) > 6 else 0)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Class Legend", fontsize=10)
    legend_path = os.path.join(out_dir, "tissue_legend.png")
    plt.savefig(legend_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {legend_path}")

    # Brain mask overlay
    brain_mask = nib.load(mask_path)
    mask_data = np.asarray(brain_mask.dataobj, dtype=np.uint8)

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
    for i, s in enumerate(slices_axial):
        ax = axes2[i]
        ax.imshow(orig_norm[s, :, :].T, cmap="gray", origin="lower", aspect="auto")
        mask_slice = mask_data[s, :, :].T.astype(bool)
        contour = mask_slice.astype(float)
        from scipy.ndimage import sobel
        edge = np.abs(sobel(contour.astype(float), axis=0)) + \
               np.abs(sobel(contour.astype(float), axis=1))
        edge = (edge > 0.1)
        ax.imshow(np.ma.masked_where(~edge, np.ones_like(edge)), cmap="Reds",
                  origin="lower", aspect="auto", alpha=0.5, interpolation="nearest")
        ax.set_title(f"Axial z={s}/{D}", fontsize=10)
        ax.axis("off")
    plt.suptitle(f"Skull Stripping — Brain Mask ({modality})", fontsize=14, y=0.98)
    mask_path_out = os.path.join(out_dir, f"brain_mask_overlay_{modality}.png")
    plt.savefig(mask_path_out, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {mask_path_out}")

    return overlay_path, mask_path_out


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base)
    out = os.path.join(project_root, "example_output")

    tasks = [
        ("sub-01_T1w.nii.gz",
         "sub-01_T1w_brain_mask.nii.gz",
         "sub-01_T1w_tissue_seg.nii.gz", "T1w"),
        ("sub-032144_ses-001_run-1_T2w.nii.gz",
         "sub-032144_ses-001_run-1_T2w_brain_mask.nii.gz",
         "sub-032144_ses-001_run-1_T2w_tissue_seg.nii.gz", "T2w"),
        ("T2_FLAIR.nii.gz",
         "T2_FLAIR_brain_mask.nii.gz",
         "T2_FLAIR_tissue_seg.nii.gz", "FLAIR"),
    ]

    for img, mask, seg, modality in tasks:
        make_overlay(
            orig_path=os.path.join(base, "example", img),
            mask_path=os.path.join(out, mask),
            seg_path=os.path.join(out, seg),
            out_dir=base,
            modality=modality,
        )
