"""
loss.py - v7 SIMPLE STABLE
────────────────────────────
Weighted BCE + Dice only.
No Lovász (NaN risk), no Focal (instability), no Boundary (complexity).
This combination is proven stable and effective on LEVIR-CD.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=3.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        pw = torch.tensor(self.pos_weight,
                          device=logits.device,
                          dtype=torch.float32)
        return F.binary_cross_entropy_with_logits(
            logits.float(), targets.float(), pos_weight=pw)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits.float()).view(-1)
        tgt_f = targets.float().view(-1)
        inter = (probs * tgt_f).sum()
        return 1 - (2 * inter + self.smooth) / (
            probs.sum() + tgt_f.sum() + self.smooth)


class LEVIRXNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce  = WeightedBCELoss(pos_weight=CFG.POS_WEIGHT)
        self.dice = DiceLoss(smooth=1.0)
        self.mse  = nn.MSELoss()

    def forward(self, change_map, change_inten,
                gt_mask, gt_intensity):
        change_map   = change_map.float()
        change_inten = change_inten.float()
        gt_intensity = gt_intensity.float()
        change_map   = torch.clamp(change_map, -15.0, 15.0)

        l_bce  = torch.nan_to_num(
            self.bce(change_map, gt_mask),  nan=0.0)
        l_dice = torch.nan_to_num(
            self.dice(change_map, gt_mask), nan=0.0)
        l_seg  = l_bce + l_dice

        inten_mean = change_inten.mean(dim=[1, 2])
        l_reg = torch.nan_to_num(
            self.mse(inten_mean, gt_intensity), nan=0.0)

        return l_seg, l_seg.item(), l_reg.item()