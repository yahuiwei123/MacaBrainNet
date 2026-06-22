"""
Custom NiftiPatchDataset — no MONAI dependency.

Each __getitem__ loads one .nii.gz volume, preprocesses it,
samples num_samples random patches, applies augmentations,
and returns stacked [num_samples, C, D, H, W] tensors.
"""

import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple, Dict, List, Optional, Callable

from data.transforms import (
    load_nifti, ensure_channel_first, reorient_to_ras,
    resample_to_spacing, clip_and_normalize, spatial_pad,
    random_foreground_crop,
    rand_flip, rand_rotate_90, rand_affine_3d,
    rand_gaussian_noise, rand_adjust_contrast, rand_scale_intensity,
    rand_bias_field, rand_gaussian_smooth, rand_invert_intensity,
    apply_brain_mask, all_flips_and_rotations,
    rand_epi_augment,
)
from data.utils import sparse_to_contig


class NiftiPatchDataset(Dataset):
    """
    Dataset that loads NIfTI volumes and returns random patches.

    Args:
        json_path: Path to split JSON (dict with 'training'/'validation' lists)
        split: 'training' or 'validation'
        patch_size: (D, H, W) — 128 for 0.5mm brain
        num_samples: number of patches per file (stacked to batch dim)
        spacing: target voxel spacing in mm
        label_ids: list of label IDs to map to 0..K-1
        aug_params: dict of augmentation parameters
        brain_mask_training: if True/1.0 always apply, if False/0.0 never apply,
            if float (e.g. 0.5) apply brain mask with that probability.
            This simulates pre-stripped inputs at inference time.
        brain_mask_label_ids: original label IDs used to build brain mask
    """

    def __init__(
        self,
        json_path: str,
        split: str = 'training',
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        num_samples: int = 4,
        spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        label_ids: Optional[List[int]] = None,
        aug_params: Optional[Dict] = None,
        brain_mask_training: float = 0.0,
        brain_mask_label_ids: Optional[List[int]] = None,
        preprocessed: bool = False,
        data_root: str = '',
    ):
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.spacing = spacing
        self.label_ids = label_ids or [0, 1]
        self.aug_params = aug_params or {}
        # Support bool (backward compat) and float probability
        if isinstance(brain_mask_training, bool):
            self.brain_mask_prob = 1.0 if brain_mask_training else 0.0
        else:
            self.brain_mask_prob = float(brain_mask_training)
        self.brain_mask_label_ids = brain_mask_label_ids
        self.preprocessed = preprocessed
        self.data_root = data_root
        self.is_train = (split == 'training')

        # Load file list from JSON
        with open(json_path, 'r') as f:
            data = json.load(f)

        key = 'training' if split == 'training' else 'validation'
        self.files = data.get(key, [])

        if len(self.files) == 0:
            raise ValueError(f"No files found for split '{key}' in {json_path}")

    def __len__(self):
        return len(self.files)

    def _preprocess_volume(self, file_info: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess a single volume."""
        img_path = os.path.join(self.data_root, file_info['image']) if self.data_root else file_info['image']
        label_path = os.path.join(self.data_root, file_info['label']) if self.data_root else file_info['label']

        # Load
        img_data, img_affine = load_nifti(img_path)
        label_data, label_affine = load_nifti(label_path)

        # Channel first
        img_data = ensure_channel_first(img_data)   # [1,D,H,W]
        label_data = ensure_channel_first(label_data)  # [1,D,H,W]

        if self.preprocessed:
            # Data already at target resolution, RAS-oriented, labels contiguous.
            # Only normalize image (skip resample, reorient, label remap).
            img_data = np.ascontiguousarray(img_data)
            label_data = np.ascontiguousarray(label_data)
            img_data = clip_and_normalize(img_data, lower_pct=0.25, upper_pct=99.75)
        else:
            # Full preprocessing pipeline
            # Reorient to RAS
            img_data, img_affine = reorient_to_ras(img_data, img_affine)
            label_data, _ = reorient_to_ras(label_data, label_affine)

            # Resample to target spacing
            img_data, img_affine = resample_to_spacing(img_data, img_affine, self.spacing, order=3,
                                                        is_label=False)
            label_data, _ = resample_to_spacing(label_data, img_affine, self.spacing, order=0,
                                                 is_label=True)

            # Ensure contiguous
            img_data = np.ascontiguousarray(img_data)
            label_data = np.ascontiguousarray(label_data)

            # Normalize image
            img_data = clip_and_normalize(img_data, lower_pct=0.25, upper_pct=99.75)

            # Label encoding: sparse → contiguous (for tissue seg with original labels)
            if self.brain_mask_prob == 0.0 and self.label_ids is not None and len(self.label_ids) > 2:
                label_data = sparse_to_contig(label_data, self.label_ids, allow_missing=True)
        
        if self.is_train:
            # Volume-level probabilistic brain mask: simulates pre-stripped inputs.
            # The entire volume is either masked or not — avoids half-masked patches.
            if self.brain_mask_prob > 0 and random.random() < self.brain_mask_prob:
                img_data = apply_brain_mask(img_data, label_data, margin=0, fill_value='min')

            # # Volume-level EPI augmentation (geometric distortion + dropout + smooth).
            # # Applied on the full volume before patch extraction — more realistic than patch-level.
            # img_t = torch.from_numpy(np.ascontiguousarray(img_data)).float()
            # label_t = torch.from_numpy(np.ascontiguousarray(label_data)).long()
            # img_t, label_t = rand_epi_augment(
            #     img_t, label_t,
            #     prob=self.aug_params.get('epi_prob', 0.4),
            # )
            # img_data = img_t.numpy()
            # label_data = label_t.numpy()

        # Pad to at least patch_size
        img_data = spatial_pad(img_data, self.patch_size, mode='minimum')
        label_data = spatial_pad(label_data, self.patch_size, mode='constant', constant_values=0.0)

        # label to int64
        label_data = label_data.astype(np.int64)

        return img_data, label_data

    def _sample_patch(self, img: np.ndarray, label: np.ndarray
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly crop a patch and apply augmentations."""
        # Random crop
        img_p, label_p, _ = random_foreground_crop(
            img, label, self.patch_size,
            pos_weight=4.0,
            neg_weight=3.0,
            edge_weight=3.0,
            edge_width=1,
            allow_smaller=True,
        )

        # Ensure patch size by padding
        img_p = spatial_pad(img_p, self.patch_size, mode='minimum')
        label_p = spatial_pad(label_p, self.patch_size, mode='constant', constant_values=0.0)

        # Convert to torch
        img_t = torch.from_numpy(img_p).float()
        label_t = torch.from_numpy(label_p).long()

        if not self.is_train:
            return img_t, label_t

        # ---- Augmentations ----

        # Intensity augmentations
        img_t = rand_invert_intensity(img_t, prob=self.aug_params.get('invert_prob', 0.5))
        img_t = rand_adjust_contrast(img_t, prob=self.aug_params.get('contrast_prob', 0.2),
                                      gamma_range=(0.7, 1.4))
        img_t = rand_scale_intensity(img_t, prob=self.aug_params.get('scale_prob', 0.2),
                                      factor=0.3)
        img_t = rand_bias_field(img_t, prob=self.aug_params.get('bias_prob', 0.5),
                                 coeff_range=(0.1, 0.6), degree=3)
        img_t = rand_gaussian_noise(img_t, prob=self.aug_params.get('noise_prob', 0.3),
                                     std=0.2)
        img_t = rand_gaussian_smooth(img_t, prob=self.aug_params.get('smooth_prob', 0.1),
                                      sigma_range=(0.15, 0.5))

        # Spatial augmentations (applied to both img and label)
        if self.aug_params.get('spatial_prob', 0.15) > 0:
            img_t, label_t = rand_affine_3d(
                img_t, label_t,
                prob=self.aug_params.get('spatial_prob', 0.15),
                rotate_range=self.aug_params.get('rotate_range', 0.3),
                shear_range=self.aug_params.get('shear_range', 0.1),
                scale_range=self.aug_params.get('scale_range', 0.1),
                translate_range=self.aug_params.get('translate_range', 8),
            )

        # Flips and 90-degree rotations
        img_t, label_t = all_flips_and_rotations(
            img_t, label_t,
            prob_flip=0.5,
            prob_rot=0.5,
        )

        return img_t, label_t

    def _validate_patch(self, img: np.ndarray, label: np.ndarray
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validation: center crop, no augmentation."""
        # Center crop (or use whole volume if smaller)
        D, H, W = img.shape[-3:]
        pD, pH, pW = self.patch_size
        d0 = max(0, (D - pD) // 2)
        h0 = max(0, (H - pH) // 2)
        w0 = max(0, (W - pW) // 2)

        slices = (
            slice(None),
            slice(d0, d0 + pD),
            slice(h0, h0 + pH),
            slice(w0, w0 + pW),
        )
        img_p = img[slices]
        label_p = label[slices]

        # Pad if smaller
        img_p = spatial_pad(img_p, self.patch_size, mode='minimum')
        label_p = spatial_pad(label_p, self.patch_size, mode='constant')

        return torch.from_numpy(img_p).float(), torch.from_numpy(label_p).long()

    def __getitem__(self, idx: int):
        file_info = self.files[idx % len(self.files)]

        # Load and preprocess
        try:
            img_vol, label_vol = self._preprocess_volume(file_info)
        except Exception as e:
            print(f"[WARN] Failed to preprocess {file_info.get('image', 'N/A')}: {e}")
            # Fallback: try the next file
            return self.__getitem__((idx + 1) % len(self.files))

        # Sample patches
        imgs, labels = [], []
        for _ in range(self.num_samples):
            if self.is_train:
                img_p, label_p = self._sample_patch(img_vol, label_vol)
            else:
                img_p, label_p = self._validate_patch(img_vol, label_vol)
            imgs.append(img_p)
            labels.append(label_p)

        # Stack: [num_samples, C, D, H, W]
        batch_img = torch.stack(imgs, dim=0)
        batch_label = torch.stack(labels, dim=0)

        # Sanity check: image should be single-channel for MRI
        if batch_img.shape[1] != 1:
            raise RuntimeError(
                f"IMAGE has wrong channels: shape={tuple(batch_img.shape)} "
                f"file={file_info['image']} volumes: img={img_vol.shape} lbl={label_vol.shape}"
            )
        if batch_label.shape[1] != 1:
            raise RuntimeError(
                f"LABEL has wrong channels: shape={tuple(batch_label.shape)} "
                f"file={file_info['image']} volumes: img={img_vol.shape} lbl={label_vol.shape}"
            )

        return {'image': batch_img, 'label': batch_label}


def build_dataloaders(
    json_path: str,
    patch_size: Tuple[int, int, int],
    num_samples: int,
    spacing: Tuple[float, float, float],
    label_ids: List[int],
    batch_size: int = 1,
    num_workers: int = 4,
    is_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
    aug_params: Optional[Dict] = None,
    brain_mask_training: float = 0.0,
    brain_mask_label_ids: Optional[List[int]] = None,
    preprocessed: bool = False,
    data_root: str = '',
    mode: str = 'cache',
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Args:
        mode: 'cache' or 'nocache' (for future extensions)
        preprocessed: if True, skip resample + label remap (data already at target resolution+indexing)
        data_root: prepended to relative paths in JSON
    """

    train_ds = NiftiPatchDataset(
        json_path=json_path,
        split='training',
        patch_size=patch_size,
        num_samples=num_samples,
        spacing=spacing,
        label_ids=label_ids,
        aug_params=aug_params,
        brain_mask_training=brain_mask_training,
        brain_mask_label_ids=brain_mask_label_ids,
        preprocessed=preprocessed,
        data_root=data_root,
    )

    val_ds = NiftiPatchDataset(
        json_path=json_path,
        split='validation',
        patch_size=patch_size,
        num_samples=num_samples,
        spacing=spacing,
        label_ids=label_ids,
        aug_params=None,
        brain_mask_training=0.0,   # val: deterministic, no random brain masking
        brain_mask_label_ids=brain_mask_label_ids,
        preprocessed=preprocessed,
        data_root=data_root,
    )

    if is_ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                           shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank,
                                         shuffle=False, drop_last=False)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, sampler=val_sampler,
            num_workers=max(1, num_workers // 2), pin_memory=True,
            persistent_workers=False,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=max(1, num_workers // 2), pin_memory=True,
            persistent_workers=False,
        )

    return train_loader, val_loader
