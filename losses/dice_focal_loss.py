"""
DiceLoss, FocalLoss, DiceFocalLoss, DiceCELoss — custom implementations.
No MONAI dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Multi-class soft Dice loss.

    Args:
        to_onehot_y: if True, convert target from class indices to one-hot
        softmax: if True, apply softmax to logits before computing Dice
        smooth: Laplace smoothing
        include_background: if False, exclude class 0 (background) from loss
        weight: class weights tensor [num_classes]
    """

    def __init__(
        self,
        to_onehot_y: bool = True,
        softmax: bool = True,
        smooth: float = 1e-5,
        include_background: bool = True,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.smooth = smooth
        self.include_background = include_background
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W] logits
            target: [B, 1, D, H, W] class indices or [B, C, D, H, W] one-hot
        """
        if self.softmax:
            pred = F.softmax(pred, dim=1)

        if self.to_onehot_y and target.shape[1] == 1:
            target = target.squeeze(1)  # [B, D, H, W]
            target_onehot = F.one_hot(target.long(), num_classes=pred.shape[1])
            target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]
        else:
            target_onehot = target.float()

        num_classes = pred.shape[1]

        if self.include_background:
            classes = range(num_classes)
        else:
            classes = range(1, num_classes)

        dice_sum = 0.0
        weight_sum = 0.0
        for c in classes:
            pred_c = pred[:, c]
            target_c = target_onehot[:, c]

            # sum over spatial dims only (all dims except batch)
            spatial_dims = tuple(range(1, pred_c.ndim))
            intersection = (pred_c * target_c).sum(dim=spatial_dims)  # [B]
            union = pred_c.sum(dim=spatial_dims) + target_c.sum(dim=spatial_dims)  # [B]

            dice_c = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_c = dice_c.mean()  # mean over batch

            w = self.weight[c] if self.weight is not None else 1.0
            dice_sum += w * dice_c
            weight_sum += w

        if weight_sum > 0:
            dice_sum = dice_sum / weight_sum

        return 1.0 - dice_sum


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Args:
        alpha: class weights (scalar or [num_classes])
        gamma: focusing parameter
        reduction: 'mean' or 'sum'
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W] logits
            target: [B, 1, D, H, W] class indices (long)
        """
        # target in class index form
        if target.shape[1] == 1:
            target = target.squeeze(1)  # [B, D, H, W]
        target = target.long()

        num_classes = pred.shape[1]
        log_p = F.log_softmax(pred, dim=1)
        log_p = log_p.permute(0, 2, 3, 4, 1).reshape(-1, num_classes)  # [N, C]
        target_flat = target.reshape(-1)  # [N]

        ce_loss = F.nll_loss(log_p, target_flat, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal Loss.

    Loss = lambda_dice * DiceLoss + lambda_focal * FocalLoss
    """

    def __init__(
        self,
        to_onehot_y: bool = True,
        softmax: bool = True,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[torch.Tensor] = None,
        include_background: bool = True,
    ):
        super().__init__()
        self.dice = DiceLoss(
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            include_background=include_background,
        )
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice(pred, target)
        focal = self.focal(pred, target)
        return self.lambda_dice * dice + self.lambda_focal * focal


class DiceCELoss(nn.Module):
    """
    Combined Dice + Cross Entropy Loss for multi-class tissue segmentation.

    Loss = lambda_dice * DiceLoss + lambda_ce * CrossEntropyLoss
    """

    def __init__(
        self,
        to_onehot_y: bool = True,
        softmax: bool = True,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        weight: Optional[torch.Tensor] = None,
        include_background: bool = True,
    ):
        super().__init__()
        self.dice = DiceLoss(
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            include_background=include_background,
            weight=weight,
        )
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape[1] == 1:
            target_squeezed = target.squeeze(1).long()
        else:
            target_squeezed = target.long()

        dice = self.dice(pred, target)
        ce = self.ce(pred, target_squeezed)
        return self.lambda_dice * dice + self.lambda_ce * ce
