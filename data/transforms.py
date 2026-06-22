"""
Custom 3D medical image transforms for training.
No MONAI dependency — all operations on numpy arrays or torch tensors.
"""

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy.ndimage import affine_transform, map_coordinates, binary_erosion, binary_dilation, zoom
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, Sequence, Optional, List, Dict
import random


# ==============================================================================
# IO & Basic Transforms
# ==============================================================================

def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file, return (data, affine)."""
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, img.affine


def ensure_channel_first(data: np.ndarray) -> np.ndarray:
    """Add channel dim if missing: [D,H,W] → [1,D,H,W]."""
    if data.ndim == 3:
        return data[np.newaxis, ...]
    return data


def reorient_to_ras(data: np.ndarray, affine: np.ndarray,
                    target_axcodes=('R', 'A', 'S')) -> Tuple[np.ndarray, np.ndarray]:
    """Reorient data to target orientation (default RAS).
    Handles both 3D [D,H,W] and 4D [C,D,H,W] channel-first arrays.
    """
    orig_ornt = nib.orientations.io_orientation(affine)
    target_ornt = nib.orientations.axcodes2ornt(target_axcodes)
    transform = nib.orientations.ornt_transform(orig_ornt, target_ornt)
    if np.allclose(transform, [[0, 1], [1, 1], [2, 1]]):
        return data, affine  # already RAS

    # nibabel apply_orientation works on first 3 axes.
    # For 4D channel-first [C,D,H,W], temporarily squeeze to 3D to avoid
    # treating the channel dim as spatial.
    if data.ndim == 4:
        spatial = data[0]  # [D,H,W]
        spatial_r = nib.orientations.apply_orientation(spatial, transform)
        data_r = spatial_r[np.newaxis, ...].astype(data.dtype)
    else:
        data_r = nib.orientations.apply_orientation(data, transform)

    affine_r = affine @ nib.orientations.inv_ornt_aff(transform, data.shape[-3:])
    return data_r, affine_r


def resample_to_spacing(data: np.ndarray, affine: np.ndarray,
                        target_spacing: Sequence[float],
                        order: int = 3,
                        is_label: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample data to target voxel spacing.
    0.5mm → [0.5, 0.5, 0.5].

    Args:
        data: [C,D,H,W] or [D,H,W]
        affine: 4x4 affine matrix
        target_spacing: (dz, dy, dx) in mm
        order: interpolation order (3= cubic for images, 0=nearest for labels)
        is_label: if True, force nearest interpolation
    """
    if is_label:
        order = 0

    current_spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    zoom_factors = current_spacing / np.array(target_spacing, dtype=float)

    if np.allclose(zoom_factors, 1.0, atol=1e-3):
        return data, affine

    spatial_shape = data.shape[-3:] if data.ndim == 4 else data.shape
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] @ np.diag(1.0 / zoom_factors)

    if data.ndim == 4:
        resampled = np.stack([
            zoom(data[c], zoom_factors, order=order, mode='reflect' if not is_label else 'nearest')
            for c in range(data.shape[0])
        ], axis=0)
    else:
        resampled = zoom(data, zoom_factors, order=order, mode='reflect' if not is_label else 'nearest')

    return resampled.astype(data.dtype), new_affine


def clip_and_normalize(data: np.ndarray, lower_pct=0.25, upper_pct=99.75) -> np.ndarray:
    """
    Robust intensity normalization:
    1. Clip to [lower_pct, upper_pct] percentiles
    2. Z-score normalize (subtract mean, divide by std)
    3. Only normalize non-zero voxels
    """
    if data.ndim == 4:
        for c in range(data.shape[0]):
            ch = data[c]
            nonzero = ch > 1e-8
            if nonzero.any():
                v = ch[nonzero]
                lo = np.percentile(v, lower_pct)
                hi = np.percentile(v, upper_pct)
                ch = np.clip(ch, lo, hi)
                mu = ch[nonzero].mean()
                std = ch[nonzero].std()
                if std > 1e-8:
                    ch = (ch - mu) / std
                ch[~nonzero] = ch[nonzero].min()
                data[c] = ch
        return data
    else:
        nonzero = data > 1e-8
        if nonzero.any():
            v = data[nonzero]
            lo = np.percentile(v, lower_pct)
            hi = np.percentile(v, upper_pct)
            data = np.clip(data, lo, hi)
            mu = data[nonzero].mean()
            std = data[nonzero].std()
            if std > 1e-8:
                data = (data - mu) / std
            data[~nonzero] = data[nonzero].min()
        return data


def spatial_pad(data: np.ndarray, target_size: Tuple[int, ...],
                mode: str = 'constant', constant_values: float = 0.0) -> np.ndarray:
    """
    Pad spatial dimensions to at least target_size.
    data: [C,D,H,W] or [D,H,W]
    """
    is_4d = (data.ndim == 4)
    spatial = data.shape[-3:] if is_4d else data.shape
    target = np.array(target_size)

    pad_width = []
    if is_4d:
        pad_width.append((0, 0))
    for s, t in zip(spatial, target):
        diff = max(0, int(t) - int(s))
        before = diff // 2
        after = diff - before
        pad_width.append((before, after))

    if np.all(np.array(pad_width[-3:]) == 0):
        return data

    if mode == 'minimum':
        constant_values = data.min()
        mode = 'constant'

    return np.pad(data, pad_width, mode=mode, constant_values=constant_values)


# ==============================================================================
# Patch Sampling
# ==============================================================================

def _compute_edge_mask(label_spatial: np.ndarray, edge_width: int = 1) -> np.ndarray:
    """Compute edge mask from label using morphological dilation - erosion."""
    fg = (label_spatial > 0)
    if not fg.any():
        return np.zeros_like(fg, dtype=bool)
    eroded = binary_erosion(fg, iterations=edge_width) if edge_width > 0 else fg
    dilated = binary_dilation(fg, iterations=edge_width) if edge_width > 0 else fg
    edge = dilated & (~eroded) & fg
    return edge


def random_foreground_crop(img: np.ndarray, label: np.ndarray,
                           patch_size: Tuple[int, int, int],
                           pos_weight: float = 4.0,
                           neg_weight: float = 3.0,
                           edge_weight: float = 3.0,
                           edge_width: int = 1,
                           allow_smaller: bool = True) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """
    Weighted random crop: edge > foreground > background.

    Args:
        img: [C,D,H,W]
        label: [1,D,H,W] or [D,H,W]
        patch_size: (D,H,W)

    Returns:
        cropped_img, cropped_label, crop_start (d0,h0,w0)
    """
    spatial_shape = np.array(img.shape[-3:])
    patch = np.array(patch_size)
    half = patch // 2

    # Valid center range so crop fits in volume
    low = half
    high = spatial_shape - patch + half
    if (high < low).any():
        high = np.maximum(low, high)

    # Ensure label is 3D spatial [D,H,W]
    if label.ndim == 4 and label.shape[0] == 1:
        label_spatial = label[0]
    elif label.ndim == 4:
        label_spatial = label[0]  # take first channel
    else:
        label_spatial = label
    label_spatial = np.squeeze(label_spatial)
    if label_spatial.ndim != 3:
        raise ValueError(f"Expected 3D label_spatial, got shape {label_spatial.shape}")

    edge_mask = _compute_edge_mask(label_spatial, edge_width)
    fg_mask = (label_spatial > 0)
    inner_fg_mask = fg_mask & (~edge_mask)

    edge_coords = np.argwhere(edge_mask)
    inner_fg_coords = np.argwhere(inner_fg_mask)

    total_w = edge_weight + pos_weight + neg_weight
    p_edge = edge_weight / total_w if len(edge_coords) > 0 else 0.0
    p_pos = pos_weight / total_w if len(inner_fg_coords) > 0 else 0.0

    r = random.random()

    if len(edge_coords) > 0 and r < p_edge:
        idx = random.randint(0, len(edge_coords) - 1)
        center = edge_coords[idx].astype(np.int64)
    elif len(inner_fg_coords) > 0 and r < (p_edge + p_pos):
        idx = random.randint(0, len(inner_fg_coords) - 1)
        center = inner_fg_coords[idx].astype(np.int64)
    else:
        # Random center from valid range
        center = np.array([random.randint(int(low[d]), int(high[d]))
                          for d in range(3)], dtype=np.int64)

    center = np.clip(center, low, high)

    start = center - half
    end = start + patch

    if allow_smaller:
        start = np.maximum(start, 0)
        end = np.minimum(end, spatial_shape)
    else:
        start = np.clip(start, 0, spatial_shape - patch)
        end = start + patch

    slices = tuple(slice(int(s), int(e)) for s, e in zip(start, end))
    img_crop = img[(slice(None),) + slices]
    label_crop = label[(slice(None),) + slices] if label.ndim == 4 else label[slices]

    return img_crop, label_crop, tuple(start)


# ==============================================================================
# Spatial Augmentations
# ==============================================================================

def rand_flip(img: torch.Tensor, label: torch.Tensor,
              prob: float = 0.5, spatial_axis: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random flip along a spatial axis. img/label: [C,D,H,W]."""
    if random.random() < prob:
        # spatial_axis is 0,1,2 for D,H,W, add 1 for channel dim
        dim = spatial_axis + 1
        img = torch.flip(img, [dim])
        label = torch.flip(label, [dim])
    return img, label


def rand_rotate_90(img: torch.Tensor, label: torch.Tensor,
                   prob: float = 0.5, max_k: int = 3,
                   spatial_axes: Tuple[int, int] = (1, 2)) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random 90-degree rotation in a plane. Default: (H,W) = axes (1,2) → (2,3) with channel."""
    if random.random() < prob:
        k = random.randint(0, max_k)
        dims = [ax + 1 for ax in spatial_axes]  # add channel offset
        img = torch.rot90(img, k, dims)
        label = torch.rot90(label, k, dims)
    return img, label


def rand_affine_3d(img: torch.Tensor, label: torch.Tensor,
                   prob: float = 0.25,
                   rotate_range: float = 0.75,
                   shear_range: float = 0.15,
                   scale_range: float = 0.15,
                   translate_range: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Random 3D affine transform using grid_sample.
    Image padding uses minimum value (shift→zeros→unshift).
    Label padding uses zeros (background).

    Args:
        img: [C,D,H,W] float32
        label: [C,D,H,W] int64
    Returns transformed tensors.
    """
    if random.random() >= prob:
        return img, label

    D, H, W = img.shape[1], img.shape[2], img.shape[3]
    device = img.device
    dtype = img.dtype

    # Build random params
    rx = random.uniform(-rotate_range, rotate_range)
    ry = random.uniform(-rotate_range, rotate_range)
    rz = random.uniform(-rotate_range, rotate_range)
    sx = random.uniform(1 - scale_range, 1 + scale_range)
    sy = random.uniform(1 - scale_range, 1 + scale_range)
    sz = random.uniform(1 - scale_range, 1 + scale_range)
    shx = random.uniform(-shear_range, shear_range)
    shy = random.uniform(-shear_range, shear_range)
    shz = random.uniform(-shear_range, shear_range)
    tx = random.uniform(-translate_range, translate_range)
    ty = random.uniform(-translate_range, translate_range)
    tz = random.uniform(-translate_range, translate_range)

    def rot_mat_x(a):
        c, s = np.cos(a), np.sin(a)
        return torch.tensor([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], device=device, dtype=torch.float32)

    def rot_mat_y(a):
        c, s = np.cos(a), np.sin(a)
        return torch.tensor([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], device=device, dtype=torch.float32)

    def rot_mat_z(a):
        c, s = np.cos(a), np.sin(a)
        return torch.tensor([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=device, dtype=torch.float32)

    scale_mat = torch.tensor([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]], device=device, dtype=torch.float32)
    shear_mat = torch.tensor([[1, shx, shy, 0], [0, 1, shz, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=device, dtype=torch.float32)
    trans_mat = torch.tensor([[1, 0, 0, tx / (D / 2)], [0, 1, 0, ty / (H / 2)], [0, 0, 1, tz / (W / 2)], [0, 0, 0, 1]], device=device, dtype=torch.float32)

    aff = rot_mat_x(rx) @ rot_mat_y(ry) @ rot_mat_z(rz) @ shear_mat @ scale_mat @ trans_mat
    aff = aff[:3, :]  # [3,4]

    # Build grid
    aff_grid = F.affine_grid(aff.unsqueeze(0), [1, 1, D, H, W], align_corners=False)

    # Image: shift min→0 so zeros-padding equals min-value padding
    img_min = img.min()
    img_shifted = img - img_min
    img_t = F.grid_sample(img_shifted.unsqueeze(0).float(), aff_grid, mode='bilinear',
                           padding_mode='zeros', align_corners=False)
    img_t = img_t + img_min

    # Label: nearest with zeros padding (background=0 is the natural minimum)
    label_t = F.grid_sample(label.unsqueeze(0).float(), aff_grid, mode='nearest',
                             padding_mode='zeros', align_corners=False)

    return img_t[0].to(dtype), label_t[0].to(label.dtype)


def all_flips_and_rotations(img: torch.Tensor, label: torch.Tensor,
                            prob_flip: float = 0.5,
                            prob_rot: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random flips (3 axes) and random 90° rotations."""
    img, label = rand_flip(img, label, prob_flip, spatial_axis=0)
    img, label = rand_flip(img, label, prob_flip, spatial_axis=1)
    img, label = rand_flip(img, label, prob_flip, spatial_axis=2)
    img, label = rand_rotate_90(img, label, prob_rot, max_k=3, spatial_axes=(1, 2))
    return img, label


# ==============================================================================
# Intensity Augmentations
# ==============================================================================

def rand_gaussian_noise(img: torch.Tensor, prob: float = 0.3,
                        std: float = 0.2) -> torch.Tensor:
    """Additive Gaussian noise."""
    if random.random() < prob:
        noise = torch.randn_like(img) * std
        img = img + noise
    return img


def rand_adjust_contrast(img: torch.Tensor, prob: float = 0.2,
                         gamma_range: Tuple[float, float] = (0.7, 1.4)) -> torch.Tensor:
    """Gamma correction / contrast adjustment."""
    if random.random() < prob:
        gamma = random.uniform(*gamma_range)
        img_min = img.min()
        img = img - img_min + 1e-8
        img = img ** gamma
        img = img + img_min
    return img


def rand_scale_intensity(img: torch.Tensor, prob: float = 0.2,
                         factor: float = 0.3) -> torch.Tensor:
    """Randomly scale intensities by factor."""
    if random.random() < prob:
        scale = random.uniform(1 - factor, 1 + factor)
        img = img * scale
    return img


def rand_bias_field(img: torch.Tensor, prob: float = 0.5,
                    coeff_range: Tuple[float, float] = (0.1, 0.6),
                    degree: int = 3) -> torch.Tensor:
    """
    Random multiplicative bias field using low-degree polynomial.
    Simulates MRI bias field inhomogeneity.
    """
    if random.random() >= prob:
        return img

    C, D, H, W = img.shape
    device, dtype = img.device, img.dtype

    # Create smooth polynomial bias field
    coeff_scale = random.uniform(*coeff_range)
    grid_z = torch.linspace(-1, 1, D, device=device)
    grid_y = torch.linspace(-1, 1, H, device=device)
    grid_x = torch.linspace(-1, 1, W, device=device)
    zz, yy, xx = torch.meshgrid(grid_z, grid_y, grid_x, indexing='ij')

    # Random polynomial coefficients
    field = torch.ones((D, H, W), device=device, dtype=torch.float32)
    for d in range(1, degree + 1):
        for dz in range(d + 1):
            for dy in range(d - dz + 1):
                dx = d - dz - dy
                c = (random.random() * 2 - 1) * coeff_scale / (d ** 2)
                field = field + c * (zz ** dz) * (yy ** dy) * (xx ** dx)

    field = field - field.min() + 0.5
    field = field / field.max()

    img = img * field[None, ...]
    return img.to(dtype)


def rand_gaussian_smooth(img: torch.Tensor, prob: float = 0.1,
                         sigma_range: Tuple[float, float] = (0.15, 0.5)) -> torch.Tensor:
    """Random Gaussian smoothing using separable 1D convolutions."""
    if random.random() >= prob or img.ndim != 4:
        return img

    C, D, H, W = img.shape
    device, dtype = img.device, img.dtype

    sigma = random.uniform(*sigma_range)
    radius = max(1, int(3.0 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k1 = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k1 = k1 / k1.sum()

    # separable 3D blur
    x = img.unsqueeze(0)  # [1,C,D,H,W]

    for dim_idx in range(3):
        # Reshape for F.conv3d: weight [C_out, C_in_per_group, Kd, Kh, Kw]
        # groups=C, so C_out=C, C_in_per_group=1
        k = k1.view(*[1 if d != dim_idx else -1 for d in range(3)])
        w_shape = [C, 1, 1, 1, 1]
        w_shape[2 + dim_idx] = len(k1)
        w = k1.reshape(1, 1, *k.shape).expand(C, -1, -1, -1, -1)
        pad = [0, 0, 0]
        pad[2 - dim_idx] = radius
        x = F.conv3d(x, w, groups=C, padding=pad)

    return x[0].to(dtype)


def rand_invert_intensity(img: torch.Tensor, prob: float = 0.5) -> torch.Tensor:
    """Randomly invert intensities while preserving mean and std."""
    if random.random() < prob:
        orig_mean = img.mean()
        orig_std = img.std()
        min_v, max_v = img.min(), img.max()
        inverted = max_v + min_v - img
        new_mean = inverted.mean()
        new_std = inverted.std().clamp(min=1e-8)
        img = (inverted - new_mean) * (orig_std / new_std) + orig_mean
    return img


# ==============================================================================
# EPI-specific augmentations (simulate fMRI/EPI artifacts from T1w/T2w)
# ==============================================================================

def _gaussian_kernel_1d_t(sigma: float, radius: int, device, dtype):
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def rand_epi_distortion(img: torch.Tensor, label: torch.Tensor,
                        prob: float = 0.5,
                        max_displacement: float = 20.0,
                        smooth_sigma: float = 12.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate EPI geometric distortion along a random axis (phase-encoding direction).

    Generates a smooth random displacement field (simulating B0 inhomogeneity)
    and warps the image using grid_sample. Label is warped with nearest-neighbor.

    Args:
        img: [C, D, H, W] float
        label: [C, D, H, W] long
        prob: probability to apply
        max_displacement: max voxel shift along phase-encoding axis
        smooth_sigma: spatial smoothness of the distortion field (voxels)
    """
    if random.random() >= prob:
        return img, label

    C, D, H, W = img.shape
    device, dtype = img.device, img.dtype
    phase_axis = random.randint(0, 2)  # 0=D, 1=H, 2=W

    # Generate smooth B0 field by filtering white noise
    field_shape = [D, H, W]
    noise = torch.randn(*field_shape, device=device, dtype=torch.float32)

    # 3D Gaussian smooth the noise to get a smooth B0 field
    radius = max(1, int(smooth_sigma * 3 + 0.5))
    x_t = noise[None, None, ...]  # [1, 1, D, H, W]
    for dim_idx in range(3):
        sigma = smooth_sigma
        r = max(1, int(sigma * 3 + 0.5))
        k1 = _gaussian_kernel_1d_t(sigma, r, device, torch.float32)
        k_shape = [1, 1, 1, 1, 1]
        k_shape[2 + dim_idx] = len(k1)
        w = k1.view(*k_shape[2:])  # 3D kernel
        w = w[None, None, ...]  # [1, 1, Kd, Kh, Kw]
        pad = [0, 0, 0]
        pad[2 - dim_idx] = r
        x_t = F.conv3d(x_t, w, padding=pad)

    displacement_field = x_t[0, 0]  # [D, H, W]
    displacement_field = displacement_field / (displacement_field.std() + 1e-8)
    displacement_field = displacement_field * max_displacement

    # Build grid
    grid_d = torch.linspace(-1, 1, D, device=device)
    grid_h = torch.linspace(-1, 1, H, device=device)
    grid_w = torch.linspace(-1, 1, W, device=device)
    dhw = torch.stack(torch.meshgrid(grid_d, grid_h, grid_w, indexing='ij'), dim=-1)  # [D,H,W,3]

    # Add displacement along phase_axis
    voxel_size = 2.0 / (torch.tensor([D, H, W], device=device, dtype=torch.float32) - 1)
    dhw[..., phase_axis] += displacement_field * voxel_size[phase_axis]

    dhw = dhw[None, ...]  # [1, D, H, W, 3]

    img_t = F.grid_sample(img.unsqueeze(0).float(), dhw, mode='bilinear',
                           padding_mode='border', align_corners=True)
    label_t = F.grid_sample(label.unsqueeze(0).float(), dhw, mode='nearest',
                             padding_mode='border', align_corners=True)

    return img_t[0].to(dtype), label_t[0].to(label.dtype)


def rand_signal_dropout(img: torch.Tensor, prob: float = 0.3,
                        num_dropouts: int = 5,
                        max_radius: float = 25.0,
                        strength: float = 0.85) -> torch.Tensor:
    """
    Simulate EPI signal dropout (susceptibility artifacts).
    Creates random ellipsoidal regions where signal is partially lost.

    Args:
        img: [C, D, H, W] float
        prob: probability to apply
        num_dropouts: number of dropout regions
        max_radius: max radius of dropout in voxels
        strength: how much signal to suppress (0=keep, 1=total loss)
    """
    if random.random() >= prob:
        return img

    C, D, H, W = img.shape
    device, dtype = img.device, img.dtype

    mask = torch.ones(1, D, H, W, device=device, dtype=torch.float32)
    d_grid = torch.arange(D, device=device, dtype=torch.float32)
    h_grid = torch.arange(H, device=device, dtype=torch.float32)
    w_grid = torch.arange(W, device=device, dtype=torch.float32)
    dd, hh, ww = torch.meshgrid(d_grid, h_grid, w_grid, indexing='ij')

    for _ in range(num_dropouts):
        cd = random.uniform(0, D - 1)
        ch = random.uniform(0, H - 1)
        cw = random.uniform(0, W - 1)
        rd = random.uniform(max_radius * 0.3, max_radius)
        rh = random.uniform(max_radius * 0.3, max_radius)
        rw = random.uniform(max_radius * 0.3, max_radius)
        s = random.uniform(0.3, strength)
        gauss = torch.exp(-0.5 * (((dd - cd) / rd) ** 2 + ((hh - ch) / rh) ** 2 + ((ww - cw) / rw) ** 2))
        mask = torch.min(mask, 1.0 - s * gauss)

    return img * mask


def rand_epi_augment(img: torch.Tensor, label: torch.Tensor,
                     prob: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a combination of EPI-like augmentations (distortion + dropout + smooth).

    This is the main entry point for EPI simulation. Call it in _sample_patch
    after standard augmentations to simulate fMRI-style images.
    """
    if random.random() < prob:
        # Geometric distortion (strong EPI characteristic)
        img, label = rand_epi_distortion(img, label, prob=1.0,
                                          max_displacement=random.uniform(5, 16))
        # Signal dropout
        img = rand_signal_dropout(img, prob=0.6, num_dropouts=random.randint(2, 8),
                                   max_radius=random.uniform(8, 20))
        # Stronger smoothing
        sigma = random.uniform(0.3, 0.9)
        radius = max(1, int(sigma * 3 + 0.5))
        k1 = _gaussian_kernel_1d_t(sigma, radius, img.device, img.dtype)
        C = img.shape[0]
        x_t = img.unsqueeze(0)  # [1,C,D,H,W]
        for dim_idx in range(3):
            k_shape = [1, 1, 1, 1, 1]
            k_shape[2 + dim_idx] = len(k1)
            w = k1.view(*k_shape[2:])[None, None, ...].expand(C, 1, -1, -1, -1)
            pad = [0, 0, 0]
            pad[2 - dim_idx] = radius
            x_t = F.conv3d(x_t, w, groups=C, padding=pad)
        img = x_t[0].to(img.dtype)
    return img, label


# ==============================================================================
# Skull Masking (for skull-stripping training)
# ==============================================================================

def apply_brain_mask(img: np.ndarray, label: np.ndarray,
                     margin: int = 0, fill_value: str = 'min') -> np.ndarray:
    """
    Mask out non-brain voxels based on label.

    Args:
        img: [C,D,H,W] or [D,H,W]
        label: same shape as img (brain mask: 0=bg, >0=brain)
        margin: dilation radius for brain mask
        fill_value: 'min' or float
    """
    brain = (label > 0)
    # squeeze to spatial (D,H,W) for dilation, then broadcast to match img
    while brain.ndim > 3:
        brain = brain[0]
    if margin > 0:
        brain = binary_dilation(brain, iterations=margin)
    brain = brain.astype(bool)
    # broadcast to match img dimensions
    if img.ndim == brain.ndim + 1:
        brain = np.broadcast_to(brain[np.newaxis, ...], img.shape).copy()

    fill = float(img[~brain].min()) if fill_value == 'min' else float(fill_value)
    img[~brain] = fill
    return img


# ==============================================================================
# Compose
# ==============================================================================

class Compose:
    """Simple chain of callable transforms."""
    def __init__(self, transforms: List):
        self.transforms = list(transforms)

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
