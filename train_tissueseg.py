#!/usr/bin/env python3
"""
Tissue Segmentation Training — SwinUNETR + DiceCELoss.
19-class segmentation. Data is preprocessed: RAS, 0.4mm, labels 0..18 contiguous.
"""

import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from trainer import Trainer, ddp_setup, ddp_cleanup
from losses.dice_focal_loss import DiceFocalLoss


class TissueSegTrainer(Trainer):
    """Trainer for 19-class tissue segmentation."""

    def __init__(self, *args, debug_save_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_save_dir = debug_save_dir
        self._debug_batch_cnt = 0
        self._max_debug_batches = 5  # only save first 5 batches

    def _build_loss(self):
        weight = torch.as_tensor(
            self.args['dataset']['label_weights'],
            device=self.device, dtype=torch.float32,
        )
        return DiceFocalLoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=1.0,
            lambda_focal=1.0,
            focal_gamma=2.0,
            focal_alpha=weight,
            include_background=True,
        )

    def _on_batch_loaded(self, inputs, labels, batch_idx):
        """Save middle slices of loaded patches for visual inspection."""
        if self.debug_save_dir is None or not self.is_main:
            return
        if self._debug_batch_cnt >= self._max_debug_batches:
            return

        os.makedirs(self.debug_save_dir, exist_ok=True)
        num_classes = self.args['dataset']['label_ids'][-1] + 1

        # inputs/labels: [num_samples, 1, D, H, W]
        num_samples = inputs.shape[0]
        for s in range(num_samples):
            img = inputs[s, 0].cpu().numpy()   # [D, H, W]
            lbl = labels[s, 0].cpu().numpy()    # [D, H, W]
            D, H, W = img.shape

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            slice_specs = [
                (D // 2, slice(None), slice(None), 'Axial (z=mid)'),
                (slice(None), H // 2, slice(None), 'Coronal (y=mid)'),
                (slice(None), slice(None), W // 2, 'Sagittal (x=mid)'),
            ]

            for i, (dz, dy, dx, title) in enumerate(slice_specs):
                im_slice = img[dz, dy, dx]
                lb_slice = lbl[dz, dy, dx]

                axes[0, i].imshow(im_slice, cmap='gray', origin='lower')
                axes[0, i].set_title(f'Image {title}')
                axes[0, i].axis('off')

                axes[1, i].imshow(lb_slice, cmap='tab20', origin='lower',
                                  vmin=0, vmax=max(num_classes - 1, 19), interpolation='nearest')
                axes[1, i].set_title(f'Label {title}')
                axes[1, i].axis('off')

            fname = os.path.join(
                self.debug_save_dir,
                f'batch_{self._debug_batch_cnt:03d}_sample_{s}.png'
            )
            plt.savefig(fname, dpi=100, bbox_inches='tight')
            plt.close()
            unique_lbls = np.unique(lbl).tolist()
            print(f"[debug] Saved {fname}  img_range=[{img.min():.3f}, {img.max():.3f}]  "
                  f"lbl_values={unique_lbls}  lbl_range=[{lbl.min()}, {lbl.max()}]")

        self._debug_batch_cnt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwinUNETR Tissue Segmentation Training")

    train_g = parser.add_argument_group("Training")
    train_g.add_argument("--batch-size", type=int, default=1)
    train_g.add_argument("--lr", "--learning-rate", type=float, default=1e-4)
    train_g.add_argument("--min-lr", type=float, default=1e-7)
    train_g.add_argument("--max-epochs", type=int, default=500)
    train_g.add_argument("--accumulation-steps", type=int, default=4)
    train_g.add_argument("--num-workers", type=int, default=8)
    train_g.add_argument("--no-shuffle", action="store_true")
    train_g.add_argument(
        "--pretrained",
        type=str,
        default="/home/weiyahui/projects/monkey/macaBrainNet/pretrain/LocalGlobal_B_step_17000.pt",
    )
    train_g.add_argument("--resume", action="store_true")

    data_g = parser.add_argument_group("Dataset")
    data_g.add_argument("--json-path", type=str, required=True)
    data_g.add_argument("--data-root", type=str,
                         default="/home/weiyahui/projects/monkey/dataset/tissue_segmentation")
    data_g.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96])
    data_g.add_argument("--num-samples", type=int, default=4)
    data_g.add_argument("--spacing", type=float, nargs=3, default=[0.4, 0.4, 0.4])
    data_g.add_argument("--fold", type=int, default=1)

    output_g = parser.add_argument_group("Output")
    output_g.add_argument("--output-dir", type=str,
                           default="/home/weiyahui/projects/monkey/macaBrainNet_v2/swinunetr_models/tissue_segmentation")

    debug_g = parser.add_argument_group("Debug")
    debug_g.add_argument("--debug-save-dir", type=str, default=None,
                          help="If set, save loaded patch slices as PNG for visual inspection")

    cli = parser.parse_args()

    # Labels already remapped to contiguous 0..18 (19 classes)
    NUM_CLASSES = 19
    LABEL_IDS = list(range(NUM_CLASSES))  # [0, 1, 2, ..., 18]
    # Class weights: bg=1, WM=4, cortex=4, LV=1, cerebellum_WM=1, cerebellum_cortex=3, ...
    LABEL_WEIGHTS = [2, 4, 4, 1, 3, 3, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 1]

    args = {
        "train_param": {
            "batch_size": cli.batch_size,
            "is_shuffle": not cli.no_shuffle,
            "num_workers": cli.num_workers,
            "learning_rate": cli.lr,
            "min_lr": cli.min_lr,
            "deep_supervision_weights": [0.96, 0.01, 0.01, 0.01, 0.01],
            "max_epochs": cli.max_epochs,
            "accumulation_steps": cli.accumulation_steps,
            "model_path": cli.pretrained,
            "resume": cli.resume,
        },
        "dataset": {
            "json_path": cli.json_path,
            "data_root": cli.data_root,
            "patch_size": cli.patch_size,
            "num_samples": cli.num_samples,
            "spacing": cli.spacing,
            "label_ids": LABEL_IDS,
            "label_weights": LABEL_WEIGHTS,
            "brain_mask_training": False,
            "preprocessed": True,  # 数据已预处理：RAS朝向、0.4mm分辨率、标签连续0..18
            "fold_num": cli.fold,
        },
        "output_dir": cli.output_dir,
    }

    os.makedirs(args["output_dir"], exist_ok=True)

    rank, world_size, local_rank = ddp_setup()
    if rank == 0:
        print("=" * 80)
        print("Tissue Segmentation Training Configuration:")
        print(f"  LR={args['train_param']['learning_rate']:.1e}  "
              f"Epochs={args['train_param']['max_epochs']}  "
              f"BatchSize={args['train_param']['batch_size']}")
        print(f"  Accum={args['train_param']['accumulation_steps']}  "
              f"NumSamples={args['dataset']['num_samples']}  "
              f"Patch={args['dataset']['patch_size']}")
        print(f"  Spacing={args['dataset']['spacing']}  "
              f"Classes={NUM_CLASSES}")
        print(f"  JSON={args['dataset']['json_path']}  "
              f"Output={args['output_dir']}")
        print(f"  WorldSize={world_size}")
        print("=" * 80)

    try:
        trainer = TissueSegTrainer(args, rank=rank, world_size=world_size, local_rank=local_rank,
                                     debug_save_dir=cli.debug_save_dir)
        best_dice, _ = trainer.train()
        if trainer.is_main:
            print(f"Training completed. Best validation Dice: {best_dice:.4f}")
    finally:
        ddp_cleanup()
