"""
Label encoding utilities — no MONAI dependency.
"""

import numpy as np
import torch


def sparse_to_contig(label, classes, allow_missing=True, unknown_id=0):
    """
    Map sparse label values to contiguous 0..K-1.

    Args:
        label: np.ndarray or torch.Tensor, integer labels
        classes: list of label IDs to map, e.g. [0,2,3,4,...,140]
        allow_missing: if True, unknown labels become unknown_id
        unknown_id: fallback class for unknown labels

    Returns:
        remapped label (same type/dims as input)
    """
    classes = np.array(sorted(set(int(c) for c in classes)), dtype=np.int64)
    max_lab = int(classes.max())
    lut = np.full((max_lab + 1,), -1, dtype=np.int64)
    for cid, lab in enumerate(classes.tolist()):
        lut[lab] = cid

    if isinstance(label, torch.Tensor):
        lut_t = torch.as_tensor(lut, dtype=torch.long, device=label.device)
        y = label.long()
        out = lut_t[y]
        if not allow_missing and (out < 0).any():
            bad = torch.unique(y[out < 0]).cpu().tolist()
            raise ValueError(f"Unknown labels not in classes: {bad}")
        out = out.clone()
        out[out < 0] = unknown_id
        return out
    else:
        y = np.asarray(label).astype(np.int64, copy=False)
        out = lut[y]
        if not allow_missing and (out < 0).any():
            bad = np.unique(y[out < 0]).tolist()
            raise ValueError(f"Unknown labels not in classes: {bad}")
        out = out.copy()
        out[out < 0] = unknown_id
        return out


def label_to_brain_mask(label):
    """
    Merge multi-tissue labels into a binary brain mask.
    label > 0 → 1 (brain), 0 → 0 (background).
    """
    return (label > 0).astype(label.dtype) if isinstance(label, np.ndarray) else (label > 0).to(label.dtype)
