"""
Base DDP Trainer — no MONAI dependency for data/transforms.
Uses MONAI only for SwinUNETR model blocks (compiled C++/CUDA ops).
"""

import os
import gc
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nets.swinunetr import SwinUNETR
from data.dataset import build_dataloaders


# ==============================================================================
# DDP utilities
# ==============================================================================

def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        os.environ.setdefault("NCCL_NSOCKS_PERTHREAD", "4")
        os.environ.setdefault("NCCL_SOCKET_NTHREADS", "2")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=30),
        )
    return rank, world_size, local_rank


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


@torch.no_grad()
def ddp_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not ddp_is_initialized():
        return x
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= dist.get_world_size()
    return x


@torch.no_grad()
def ddp_reduce_mean_vec(x: torch.Tensor) -> torch.Tensor:
    if not ddp_is_initialized():
        return x
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= dist.get_world_size()
    return x


@torch.no_grad()
def ddp_broadcast_(x: torch.Tensor, src: int = 0):
    if ddp_is_initialized():
        dist.broadcast(x, src=src)


# ==============================================================================
# Safe checkpoint loading
# ==============================================================================

def _safe_torch_load(ckpt_path: str, map_location="cpu", trust_source: bool = False):
    """PyTorch 2.6+ compatible loading."""
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except Exception as e1:
        msg1 = str(e1)

    if "Unsupported global" in msg1 and "numpy.core.multiarray.scalar" in msg1:
        try:
            import numpy.core.multiarray as ncm
            torch.serialization.add_safe_globals([ncm.scalar])
            with torch.serialization.safe_globals([ncm.scalar]):
                return torch.load(ckpt_path, map_location=map_location, weights_only=True)
        except Exception as e2:
            msg2 = str(e2)
    else:
        msg2 = msg1

    if trust_source:
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)

    raise RuntimeError(
        f"Checkpoint load failed under weights_only=True.\n"
        f"Error: {msg2}\n"
        "Set trust_source=True only if you TRUST the checkpoint source."
    )


def _remap_localglobal_keys(state: dict) -> dict:
    """Remap LocalGlobal pretrained checkpoint keys to SwinUNETR format."""
    new_state = {}
    for k, v in state.items():
        if k.startswith("encoder.norm.") or k.startswith("projection."):
            continue
        if k.startswith("encoder.encoder.swinViT."):
            new_state[k[len("encoder.encoder."):]] = v
        elif k.startswith("encoder.encoder.layers"):
            new_state["swinViT." + k[len("encoder.encoder."):]] = v
        elif k.startswith("encoder.encoder.encoder"):
            new_state[k[len("encoder.encoder."):]] = v
        elif k.startswith("decoder.decoder"):
            new_state[k[len("decoder."):]] = v
        elif k.startswith("decoder.ds_heads."):
            new_state["out_ds." + k[len("decoder.ds_heads."):]] = v
        elif k.startswith("decoder.out_conv."):
            new_state["out.conv.conv" + k[len("decoder.out_conv."):]] = v
        else:
            new_state[k] = v
    return new_state


def load_ckpt_allow_mismatch(model: nn.Module, ckpt_path: str, verbose: bool = True,
                              trust_source: bool = False):
    """Load pretrained weights with shape-mismatch skipping."""
    ckpt = _safe_torch_load(ckpt_path, map_location="cpu", trust_source=trust_source)

    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint format unexpected: type(state)={type(state)}")

    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}

    if any(k.startswith("encoder.encoder.") for k in state.keys()):
        state = _remap_localglobal_keys(state)
        if verbose:
            print(f"[ckpt] detected LocalGlobal format, remapped keys")

    model_state = model.state_dict()
    loaded, skipped = [], []
    filtered = {}

    for k, v in model_state.items():
        if k in state and hasattr(state[k], "shape") and state[k].shape == v.shape:
            filtered[k] = state[k]
            loaded.append(k)
        else:
            filtered[k] = v
            if k in state and hasattr(state[k], "shape") and hasattr(v, "shape"):
                skipped.append((k, tuple(state[k].shape), tuple(v.shape)))

    model.load_state_dict(filtered, strict=True)

    if verbose:
        print(f"[ckpt] loaded={len(loaded)}/{len(model_state)} from {ckpt_path}")
        if skipped:
            print(f"[ckpt] skipped(shape mismatch)={len(skipped)} (showing up to 10)")
            for k, s0, s1 in skipped[:10]:
                print(f"  - {k}: ckpt{s0} != model{s1}")

    return loaded, skipped


# ==============================================================================
# Sliding Window Inference (no MONAI)
# ==============================================================================

@torch.no_grad()
def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: tuple,
    network: nn.Module,
    overlap: float = 0.20,
    mode: str = "gaussian",
    sigma_scale: float = 0.125,
    device: torch.device = None,
):
    """
    Sliding window inference for 3D volumes.
    Returns logits of shape [B, C, D, H, W].
    """
    B, C, D, H, W = inputs.shape
    pD, pH, pW = roi_size
    stride_d = int(pD * (1 - overlap))
    stride_h = int(pH * (1 - overlap))
    stride_w = int(pW * (1 - overlap))

    # Compute output channels
    dummy = torch.zeros(1, C, pD, pH, pW, device=inputs.device)
    out_channels = network(dummy)[0].shape[1]

    full_logits = torch.zeros(B, out_channels, D, H, W, device=inputs.device)
    count_map = torch.zeros(B, 1, D, H, W, device=inputs.device)

    # Gaussian importance map
    if mode == "gaussian":
        gauss = _gaussian_kernel_3d(pD, pH, pW, sigma_scale, inputs.device)
    else:
        gauss = torch.ones((1, 1, pD, pH, pW), device=inputs.device)

    d_starts = list(range(0, max(1, D - pD + 1), stride_d))
    if not d_starts or d_starts[-1] + pD < D:
        d_starts.append(max(0, D - pD))
    h_starts = list(range(0, max(1, H - pH + 1), stride_h))
    if not h_starts or h_starts[-1] + pH < H:
        h_starts.append(max(0, H - pH))
    w_starts = list(range(0, max(1, W - pW + 1), stride_w))
    if not w_starts or w_starts[-1] + pW < W:
        w_starts.append(max(0, W - pW))

    for d0 in d_starts:
        for h0 in h_starts:
            for w0 in w_starts:
                d1 = d0 + pD
                h1 = h0 + pH
                w1 = w0 + pW
                patch = inputs[:, :, d0:d1, h0:h1, w0:w1]
                raw_out = network(patch)
                if isinstance(raw_out, (list, tuple)):
                    logits = raw_out[0]
                    del raw_out  # free other deep supervision tensors
                else:
                    logits = raw_out
                full_logits[:, :, d0:d1, h0:h1, w0:w1] += logits * gauss
                count_map[:, :, d0:d1, h0:h1, w0:w1] += gauss
                del logits, patch

    full_logits = full_logits / count_map.clamp(min=1e-8)
    return full_logits


def _gaussian_kernel_3d(D, H, W, sigma_scale, device):
    d = torch.arange(D, device=device, dtype=torch.float32) - (D - 1) / 2.0
    h = torch.arange(H, device=device, dtype=torch.float32) - (H - 1) / 2.0
    w = torch.arange(W, device=device, dtype=torch.float32) - (W - 1) / 2.0
    sigma_d = D * sigma_scale
    sigma_h = H * sigma_scale
    sigma_w = W * sigma_scale
    gd = torch.exp(-0.5 * (d / sigma_d) ** 2)
    gh = torch.exp(-0.5 * (h / sigma_h) ** 2)
    gw = torch.exp(-0.5 * (w / sigma_w) ** 2)
    gauss = gd[:, None, None] * gh[None, :, None] * gw[None, None, :]
    gauss = gauss / gauss.sum()
    return gauss[None, None, ...]  # [1, 1, D, H, W]


# ==============================================================================
# Metrics
# ==============================================================================

@torch.no_grad()
def per_class_dice_from_ids(pred_ids: torch.Tensor, label_ids: torch.Tensor,
                            num_classes: int):
    """
    pred_ids: [B, 1, D, H, W] long
    label_ids: [B, 1, D, H, W] long
    Returns: dice [num_classes], present_mask [num_classes]
    """
    pred = pred_ids.view(-1)
    lab = label_ids.view(-1)
    dice = torch.zeros(num_classes, device=pred_ids.device, dtype=torch.float32)
    present = torch.zeros(num_classes, device=pred_ids.device, dtype=torch.float32)

    for c in range(num_classes):
        lab_c = (lab == c)
        if lab_c.any():
            present[c] = 1.0
            pred_c = (pred == c)
            inter = (pred_c & lab_c).sum().float()
            denom = pred_c.sum().float() + lab_c.sum().float()
            dice[c] = (2.0 * inter + 1e-5) / (denom + 1e-5)

    return dice, present


# ==============================================================================
# Trainer Class
# ==============================================================================

class Trainer:
    def __init__(self, args: dict, rank: int = 0, world_size: int = 1, local_rank: int = 0):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_ddp = world_size > 1
        self.is_main = (rank == 0)

        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        self.num_classes = len(self.args['dataset']['label_ids'])
        self.deep_supervision_weights = self.args['train_param']['deep_supervision_weights']

        # Model
        self.model = self._build_model().to(self.device)

        # DDP
        if self.is_ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                broadcast_buffers=True,
                gradient_as_bucket_view=True,
                bucket_cap_mb=200,
            )

        # Loss
        self.loss_function = self._build_loss()

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args['train_param']['learning_rate'],
            weight_decay=1e-5,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args['train_param']['max_epochs'],
            eta_min=args['train_param']['min_lr'],
        )

        # Dynamic sampling state
        self.dice_ema = torch.zeros(self.num_classes, device=self.device)
        self.present_ema = torch.zeros(self.num_classes, device=self.device)
        self.ema_m = 0.95
        self.sampling_alpha = 2.0
        self.uniform_mix = 0.2

        # Resume
        self._resumed_epoch = -1
        if args['train_param'].get('resume', False):
            self._resumed_epoch = self._load_checkpoint()

        # TensorBoard
        if self.is_main:
            log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

    # ---- subclasses can override these ----

    def _build_model(self) -> nn.Module:
        model = SwinUNETR(
            img_size=self.args['dataset']['patch_size'],
            in_channels=1,
            out_channels=self.num_classes,
            patch_size=(2, 2, 2),
            feature_size=48,
            use_v2=True,
        )
        pretrained_path = self.args['train_param'].get('model_path', '')
        if pretrained_path and pretrained_path.lower() != 'none' and not \
                self.args['train_param'].get('resume', False):
            load_ckpt_allow_mismatch(model, pretrained_path, verbose=True, trust_source=True)
        return model

    def _build_loss(self):
        """Override in subclass."""
        raise NotImplementedError

    def _init_transforms(self):
        """Override in subclass (not used directly; handled in dataset)."""
        pass

    def _on_batch_loaded(self, inputs: torch.Tensor, labels: torch.Tensor, batch_idx: int):
        """Hook called after data is loaded and before model forward.
        Override in subclass for debugging/visualization."""
        pass

    # ---- Data ----

    def _get_data_loaders(self):
        """Create train/val DataLoaders using custom (no-MONAI) dataset."""
        ds_args = self.args['dataset']

        train_loader, val_loader = build_dataloaders(
            json_path=ds_args['json_path'],
            patch_size=tuple(ds_args['patch_size']),
            num_samples=ds_args['num_samples'],
            spacing=tuple(ds_args['spacing']),
            label_ids=ds_args['label_ids'],
            num_workers=self.args['train_param']['num_workers'],
            is_ddp=self.is_ddp,
            rank=self.rank,
            world_size=self.world_size,
            aug_params=self.args.get('aug_params', None),
            brain_mask_training=ds_args.get('brain_mask_training', False),
            brain_mask_label_ids=ds_args.get('brain_mask_label_ids', None),
            preprocessed=ds_args.get('preprocessed', False),
            data_root=ds_args.get('data_root', ''),
        )
        return train_loader, val_loader

    # ---- Loss helpers ----

    def _resize_labels_to_match(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        target_size = output.shape[-3:]
        lab = labels
        if lab.dim() == 4:
            lab = lab.unsqueeze(1)
        if tuple(lab.shape[-3:]) == tuple(target_size):
            return lab
        orig_dtype = lab.dtype
        lab_rs = F.interpolate(lab.float(), size=target_size, mode="nearest")
        if orig_dtype in (torch.long, torch.int64, torch.int32, torch.uint8):
            lab_rs = lab_rs.to(orig_dtype)
        return lab_rs

    def deep_supervision_loss(self, outputs, labels, loss_function, weights=None):
        if isinstance(outputs, (list, tuple)):
            num_layers = len(outputs)
            if weights is None:
                weights = [1.0 / num_layers] * num_layers
            total_loss = 0.0
            for i, out in enumerate(outputs):
                lab_i = self._resize_labels_to_match(out, labels)
                total_loss = total_loss + weights[i] * loss_function(out, lab_i)
            return total_loss

        if isinstance(outputs, torch.Tensor) and outputs.dim() >= 6:
            num_layers = outputs.shape[1]
            if weights is None:
                weights = [1.0 / num_layers] * num_layers
            total_loss = 0.0
            for i in range(num_layers):
                out = outputs[:, i, ...]
                lab_i = self._resize_labels_to_match(out, labels)
                total_loss = total_loss + weights[i] * loss_function(out, lab_i)
            return total_loss

        if isinstance(outputs, torch.Tensor) and outputs.dim() >= 5:
            lab = self._resize_labels_to_match(outputs, labels)
            return loss_function(outputs, lab)

        return loss_function(outputs, labels)

    # ---- Checkpoint ----

    def _get_state_dict(self):
        return self.model.module.state_dict() if self.is_ddp else self.model.state_dict()

    @staticmethod
    def _safe_save(obj, filepath, max_retries=3):
        """Atomically save with unique temp name to avoid race conditions."""
        tmp_path = filepath + f".tmp.{os.getpid()}.{os.urandom(4).hex()}"
        for attempt in range(max_retries):
            try:
                torch.save(obj, tmp_path)
                os.replace(tmp_path, filepath)
                return
            except (FileNotFoundError, OSError) as e:
                if attempt == max_retries - 1:
                    raise
                # clean up stale tmp and retry
                for f in (tmp_path, filepath + ".tmp"):
                    try:
                        if os.path.exists(f):
                            os.remove(f)
                    except Exception:
                        pass
                import time
                time.sleep(1)

    def save_checkpoint(self, epoch, best_dice, filepath):
        checkpoint = {
            'model_state': self._get_state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_dice': best_dice,
            'dice_ema': self.dice_ema,
            'present_ema': self.present_ema,
            'rng_state': torch.random.get_rng_state(),
            'cuda_rng_state': torch.cuda.random.get_rng_state() if torch.cuda.is_available() else None,
        }
        self._safe_save(checkpoint, filepath)

    def _load_checkpoint(self):
        ckpt_path = os.path.join(self.args['output_dir'], "latest_checkpoint.pth")
        if not os.path.exists(ckpt_path):
            if self.is_main:
                print(f"[resume] Checkpoint not found: {ckpt_path}, starting from scratch")
            return -1

        if self.is_main:
            print(f"[resume] Loading checkpoint: {ckpt_path}")

        checkpoint = _safe_torch_load(ckpt_path, map_location="cpu", trust_source=True)
        state = checkpoint['model_state']
        if any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        model = self.model.module if self.is_ddp else self.model
        model.load_state_dict(state, strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.dice_ema = checkpoint['dice_ema'].to(self.device)
        self.present_ema = checkpoint['present_ema'].to(self.device)

        if 'rng_state' in checkpoint:
            torch.random.set_rng_state(checkpoint['rng_state'])
        if 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None \
                and torch.cuda.is_available():
            torch.cuda.random.set_rng_state(checkpoint['cuda_rng_state'])

        resumed_epoch = checkpoint['epoch']
        if self.is_main:
            print(f"[resume] Restored epoch={resumed_epoch}")

        return resumed_epoch

    # ---- Training loop ----

    def train_epoch(self, epoch, train_loader):
        self.model.train()

        if self.is_ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_dice = 0.0
        step = 0

        accum = self.args['train_param']['accumulation_steps']
        self.optimizer.zero_grad(set_to_none=True)

        iterator = tqdm(train_loader, desc=f"Train Epoch {epoch}",
                       disable=not self.is_main, unit="batch")

        for batch_data in iterator:
            step += 1
            inputs = batch_data["image"].to(self.device, non_blocking=True)
            labels = batch_data["label"].to(self.device, non_blocking=True)

            # Squeeze extra batch dim from DataLoader (batch_size=1 inside dataset already stacks num_samples)
            if inputs.dim() == 6 and inputs.shape[0] == 1:
                inputs = inputs.squeeze(0)
                labels = labels.squeeze(0)

            # labels shape check: should be [num_samples, 1, D, H, W]
            if labels.dim() == 4:
                labels = labels.unsqueeze(1)

            # Debug label range check
            with torch.no_grad():
                if labels.dim() == 5 and labels.shape[1] == self.num_classes:
                    pass  # one-hot
                else:
                    label_max = labels.max().item()
                    label_min = labels.min().item()
                    if label_max >= self.num_classes or label_min < 0:
                        raise ValueError(
                            f"Label out of range! label_min={label_min}, label_max={label_max}, "
                            f"num_classes={self.num_classes}"
                        )

            if inputs.shape[1] != 1:
                raise RuntimeError(
                    f"Expected input channel=1 but got shape={tuple(inputs.shape)}. "
                    f"labels shape={tuple(labels.shape)}, label range=[{labels.min().item()}, {labels.max().item()}]"
                )

            self._on_batch_loaded(inputs, labels, step)

            outputs = self.model(inputs)
            loss = self.deep_supervision_loss(
                outputs=outputs, labels=labels,
                loss_function=self.loss_function,
                weights=self.deep_supervision_weights,
            )

            if self.is_ddp and hasattr(self.model, "no_sync") and not \
                    (step % accum == 0 or step == len(train_loader)):
                with self.model.no_sync():
                    loss.backward()
            else:
                loss.backward()

            if step % accum == 0 or step == len(train_loader):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if step % (accum * 50) == 0:
                    torch.cuda.empty_cache()

            # Dice metrics
            out0 = outputs[0] if isinstance(outputs, (list, tuple)) else \
                (outputs[:, 0] if outputs.dim() > 5 else outputs)
            pred_ids = torch.argmax(out0, dim=1, keepdim=True).long()
            label_ids = labels.long()
            if label_ids.dim() == 5 and label_ids.shape[1] != 1:
                label_ids = torch.argmax(label_ids, dim=1, keepdim=True).long()

            dice_c, present_c = per_class_dice_from_ids(pred_ids, label_ids, self.num_classes)
            dice_c = ddp_reduce_mean_vec(dice_c)
            present_c = ddp_reduce_mean_vec(present_c)

            # EMA update
            m = self.ema_m
            if self.is_main:
                gate = (present_c > 0).float()
                self.dice_ema = self.dice_ema * (m ** gate) + dice_c * (1 - (m ** gate))
                self.present_ema = self.present_ema * m + present_c * (1 - m)

            loss_mean = ddp_reduce_mean(loss.detach())
            dice_mean = dice_c[1:].mean()

            epoch_loss += float(loss_mean.item())
            epoch_dice += float(dice_mean.item())

            if self.is_main:
                iterator.set_postfix(loss=float(loss_mean.item()), dice=float(dice_mean.item()))
                gs = epoch * len(train_loader) + step
                self.writer.add_scalar("train/loss_step", float(loss_mean.item()), gs)
                self.writer.add_scalar("train/dice_step", float(dice_mean.item()), gs)

            del batch_data, inputs, labels, outputs, out0, pred_ids, label_ids, dice_c, present_c, loss

            if step % 200 == 0:
                gc.collect()

        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        epoch_loss /= max(1, len(train_loader))
        epoch_dice /= max(1, len(train_loader))

        if self.is_main:
            self.writer.add_scalar("train/loss_epoch", epoch_loss, epoch)
            self.writer.add_scalar("train/dice_epoch", epoch_dice, epoch)

        # Dynamic sampling ratios
        new_ratios = torch.ones(self.num_classes, device=self.device) / self.num_classes
        if self.is_main:
            diff = (1.0 - self.dice_ema.clamp(0, 1)).pow(self.sampling_alpha)
            rare_mask = (self.present_ema < 0.01)
            diff = torch.where(rare_mask, torch.zeros_like(diff), diff)
            if diff.sum() > 0:
                p = diff / (diff.sum() + 1e-8)
            else:
                p = torch.ones_like(diff) / self.num_classes
            u = torch.ones_like(p) / self.num_classes
            p = (1 - self.uniform_mix) * p + self.uniform_mix * u
            p = p / p.sum()
            new_ratios = p

        ddp_broadcast_(new_ratios, src=0)

        if self.is_main:
            label_names = self.args['dataset']['label_ids']
            print("\n=== Training Per-class EMA Dice ===")
            for c in range(self.num_classes):
                d_val = float(self.dice_ema[c].item())
                p_val = float(self.present_ema[c].item())
                self.writer.add_scalar(f"train/dice_ema_class_{label_names[c]}", d_val, epoch)
                print(f"Class {label_names[c]:3d} (present: {p_val:.2f}): {d_val:.4f}")
            print("=" * 45)

        return epoch_loss, epoch_dice

    @torch.no_grad()
    def val_epoch(self, epoch, val_loader):
        self.model.eval()
        model_for_infer = self.model.module if self.is_ddp else self.model

        dice_sum = torch.zeros(1, device=self.device)
        n_sum = torch.zeros(1, device=self.device)
        per_class_dice_sum = torch.zeros(self.num_classes, device=self.device)
        per_class_count = torch.zeros(self.num_classes, device=self.device)

        if self.is_ddp and hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(epoch)

        patch_size = tuple(self.args['dataset']['patch_size'])

        iterator = tqdm(val_loader, desc=f"Val Epoch {epoch}", disable=not self.is_main)

        for batch_data in iterator:
            inputs = batch_data["image"].to(self.device, non_blocking=True)
            labels = batch_data["label"].to(self.device, non_blocking=True)

            if inputs.dim() == 6 and inputs.shape[0] == 1:
                inputs = inputs.squeeze(0)
                labels = labels.squeeze(0)
            if labels.dim() == 4:
                labels = labels.unsqueeze(1)

            logits = sliding_window_inference(
                inputs=inputs,
                roi_size=patch_size,
                network=model_for_infer,
                overlap=0.20,
            )

            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            pred_ids = torch.argmax(logits, dim=1, keepdim=True).long()
            label_ids = labels.long()
            if label_ids.dim() == 5 and label_ids.shape[1] != 1:
                label_ids = torch.argmax(label_ids, dim=1, keepdim=True).long()

            batch_dice_c, batch_present_c = per_class_dice_from_ids(
                pred_ids, label_ids, self.num_classes)
            per_class_dice_sum += batch_dice_c
            per_class_count += batch_present_c

            dice = batch_dice_c[1:].mean()
            dice_sum += dice
            n_sum += 1.0

            if self.is_main:
                curr = float(dice.item())
                mean_val = float((dice_sum / (n_sum + 1e-8)).item())
                iterator.set_postfix(dice=f"{curr:.4f}", mean=f"{mean_val:.4f}")

            del batch_data, inputs, labels, logits, pred_ids, label_ids, batch_dice_c, batch_present_c
            torch.cuda.empty_cache()

        gc.collect()

        if ddp_is_initialized():
            dist.all_reduce(dice_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(per_class_dice_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(per_class_count, op=dist.ReduceOp.SUM)

        if n_sum.item() < 1:
            if self.is_main:
                print("[WARN] All validation samples skipped.")
            return None, None

        val_dice = float((dice_sum / (n_sum + 1e-8)).item())
        per_class_dice = per_class_dice_sum / (per_class_count + 1e-8)
        per_class_dice[per_class_count == 0] = 0.0

        if self.is_main and self.writer is not None:
            self.writer.add_scalar("val/dice", val_dice, epoch)
            label_names = self.args['dataset']['label_ids']
            for c in range(self.num_classes):
                self.writer.add_scalar(f"val/dice_class_{label_names[c]}",
                                       float(per_class_dice[c].item()), epoch)

            print("\n=== Validation Per-class Dice ===")
            for c in range(self.num_classes):
                cnt = int(per_class_count[c].item())
                d_val = float(per_class_dice[c].item())
                print(f"Class {label_names[c]:3d} (samples: {cnt:3d}): {d_val:.4f}")
            print("=" * 40)

        return val_dice, per_class_dice

    def train(self):
        train_loader, val_loader = self._get_data_loaders()

        if self._resumed_epoch >= 0:
            start_epoch = self._resumed_epoch + 1
            ckpt = _safe_torch_load(
                os.path.join(self.args['output_dir'], "latest_checkpoint.pth"),
                map_location="cpu", trust_source=True)
            best_dice = ckpt.get('best_dice', 0.0)
        else:
            start_epoch = 0
            best_dice = 0.0

        val_dice = 0.0
        per_class_dice = None

        for epoch in range(start_epoch, self.args['train_param']['max_epochs']):
            train_loss, train_dice = self.train_epoch(epoch, train_loader)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            if self.is_ddp:
                dist.barrier()

            # Save
            if self.is_main:
                curr_path = os.path.join(self.args['output_dir'], "curr_3d_swinunetr_model.pth")
                self._safe_save(self._get_state_dict(), curr_path)
                self.save_checkpoint(epoch, best_dice,
                                     os.path.join(self.args['output_dir'], "latest_checkpoint.pth"))

            do_val = (epoch + 1) % 2 == 0

            if do_val:
                val_dice, per_class_dice = self.val_epoch(epoch, val_loader)
                if val_dice is None:
                    val_dice = 0.0

                if val_dice > best_dice and self.is_main:
                    best_dice = val_dice
                    best_path = os.path.join(self.args['output_dir'], "best_3d_swinunetr_model.pth")
                    self._safe_save(self._get_state_dict(), best_path)
                    print(f"New best model saved with Dice: {best_dice:.4f}")

                    if per_class_dice is not None:
                        best_dice_dict = {
                            "total_dice": best_dice,
                            "per_class_dice": {
                                str(self.args['dataset']['label_ids'][c]):
                                float(per_class_dice[c].item())
                                for c in range(self.num_classes)
                            },
                        }
                        dice_path = os.path.join(self.args['output_dir'],
                                                  "best_model_per_class_dice.pth")
                        self._safe_save(best_dice_dict, dice_path)

            print(f"Epoch {epoch + 1}/{self.args['train_param']['max_epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
                  f"Val Dice: {val_dice:.4f} | LR: {current_lr:.6e}")

            if self.is_main and self.writer is not None:
                self.writer.flush()

            gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

        if self.is_main and self.writer is not None:
            self.writer.close()
        return best_dice, per_class_dice
