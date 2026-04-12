"""
train.py - v9 DEFINITIVE FIX
──────────────────────────────
Key fixes:
1. NO AMP — full float32, eliminates TIAM softmax overflow entirely
2. AdamW with differential LR (backbone protected at 1e-5)
3. 3-stage freeze: stage1 decoder only, stage2 top encoder, stage3 all
4. Cosine annealing LR schedule
5. 5-epoch linear warmup before full LR
6. NaN check BEFORE backward
7. Tight grad clip (0.5) throughout
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from config import CFG


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics_from_counts(tp, fp, fn, tn):
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = tp / (tp + fp + fn + 1e-6)
    oa        = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    return dict(F1=float(f1), IoU=float(iou),
                Precision=float(precision),
                Recall=float(recall), OA=float(oa))


def compute_metrics(pred_logits, gt_mask, threshold=0.5):
    preds = (torch.sigmoid(pred_logits) > threshold).long()
    preds = preds.cpu().numpy().flatten()
    gts   = gt_mask.cpu().numpy().flatten()
    tp = int(((preds == 1) & (gts == 1)).sum())
    fp = int(((preds == 1) & (gts == 0)).sum())
    fn = int(((preds == 0) & (gts == 1)).sum())
    tn = int(((preds == 0) & (gts == 0)).sum())
    return compute_metrics_from_counts(tp, fp, fn, tn)


def _update_counts(probs, mask, threshold):
    preds = (probs > threshold).long().numpy().flatten()
    gts   = mask.long().numpy().flatten()
    tp = int(((preds == 1) & (gts == 1)).sum())
    fp = int(((preds == 1) & (gts == 0)).sum())
    fn = int(((preds == 0) & (gts == 1)).sum())
    tn = int(((preds == 0) & (gts == 0)).sum())
    return tp, fp, fn, tn


# ─────────────────────────────────────────────────────────────
# Optimizer builder
# ─────────────────────────────────────────────────────────────

def build_optimizer(model, stage: int):
    """
    Stage 1: Only decoder/TIAM/proj layers — backbone frozen.
             High LR safe because pretrained weights are protected.
    Stage 2: Top 30% encoder at very low LR, rest at medium LR.
    Stage 3: Full network, all at low LR.
    """
    has_encoder = hasattr(model, 'encoder')

    if stage == 1:
        trainable = [p for p in model.parameters() if p.requires_grad]
        params = [{"params": trainable, "lr": CFG.LR_HEAD}]

    elif stage == 2:
        enc_params, dec_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (enc_params if 'encoder' in name else dec_params).append(p)
        params = [
            {"params": enc_params, "lr": CFG.LR_BACKBONE},
            {"params": dec_params, "lr": CFG.LR_HEAD / 5},
        ]

    else:  # stage 3
        enc_params, dec_params = [], []
        for name, p in model.named_parameters():
            (enc_params if 'encoder' in name else dec_params).append(p)
        params = [
            {"params": enc_params, "lr": CFG.LR_BACKBONE},
            {"params": dec_params, "lr": CFG.LR_BACKBONE * 10},
        ]

    optimizer = optim.AdamW(params, weight_decay=CFG.WEIGHT_DECAY)
    for pg in optimizer.param_groups:
        pg['initial_lr'] = pg['lr']
    return optimizer


# ─────────────────────────────────────────────────────────────
# Training epoch — full float32, no AMP
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss    = 0.0
    valid_batches = 0
    tp_sum = fp_sum = fn_sum = tn_sum = 0

    # Linear warmup for first 5 epochs
    WARMUP = 5
    if epoch <= WARMUP:
        factor = epoch / WARMUP
        for pg in optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * factor

    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch_idx, (img1, img2, mask, intensity) in enumerate(pbar):

        img1      = img1.to(CFG.DEVICE)
        img2      = img2.to(CFG.DEVICE)
        mask      = mask.to(CFG.DEVICE)
        intensity = intensity.to(CFG.DEVICE)

        # Full float32 — no AMP, no float16 TIAM overflow
        cmap, cinten       = model(img1, img2)
        loss, l_seg, l_reg = criterion(cmap, cinten, mask, intensity)

        # NaN check BEFORE backward
        if not torch.isfinite(loss):
            print(f"\n  [WARN] NaN at batch {batch_idx}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        probs    = torch.sigmoid(cmap.detach()).cpu()
        mask_cpu = mask.detach().cpu()
        tp, fp, fn, tn = _update_counts(probs, mask_cpu, CFG.THRESHOLD)
        tp_sum += tp; fp_sum += fp
        fn_sum += fn; tn_sum += tn
        total_loss    += loss.item()
        valid_batches += 1

        vram = (f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
                if CFG.DEVICE.type == "cuda" else "cpu")
        pbar.set_postfix(loss=f"{loss.item():.4f}", vram=vram)

    avg_loss = total_loss / max(valid_batches, 1)
    metrics  = compute_metrics_from_counts(
        tp_sum, fp_sum, fn_sum, tn_sum)
    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss    = 0.0
    valid_batches = 0
    prob_list     = []
    mask_list     = []

    for img1, img2, mask, intensity in tqdm(
            loader, desc="  Val  ", leave=False):
        img1      = img1.to(CFG.DEVICE)
        img2      = img2.to(CFG.DEVICE)
        mask      = mask.to(CFG.DEVICE)
        intensity = intensity.to(CFG.DEVICE)

        cmap, cinten       = model(img1, img2)
        loss, l_seg, l_reg = criterion(cmap, cinten, mask, intensity)

        if torch.isfinite(loss):
            total_loss    += loss.item()
            valid_batches += 1

        prob_list.append(
            torch.sigmoid(cmap).cpu().numpy().astype(np.float16))
        mask_list.append(mask.cpu().numpy().astype(np.uint8))

    avg_loss  = total_loss / max(valid_batches, 1)
    all_probs = np.concatenate([p.flatten() for p in prob_list])
    all_masks = np.concatenate([m.flatten() for m in mask_list])

    best_thr, best_m = CFG.THRESHOLD, None
    for thr in np.linspace(0.1, 0.9, 81):
        preds = (all_probs > thr).astype(np.uint8)
        tp = int(((preds == 1) & (all_masks == 1)).sum())
        fp = int(((preds == 1) & (all_masks == 0)).sum())
        fn = int(((preds == 0) & (all_masks == 1)).sum())
        tn = int(((preds == 0) & (all_masks == 0)).sum())
        m  = compute_metrics_from_counts(tp, fp, fn, tn)
        if best_m is None or m["F1"] > best_m["F1"]:
            best_m, best_thr = m, float(thr)

    del prob_list, mask_list, all_probs, all_masks
    return avg_loss, best_m, best_thr


# ─────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, criterion):

    optimizer = build_optimizer(model, stage=1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.EPOCHS, eta_min=1e-7)

    best_f1        = 0.0
    best_threshold = CFG.THRESHOLD
    patience_c     = 0
    start_epoch    = 1
    current_stage  = 1

    history = {"train_loss": [], "val_loss": [],
               "val_f1": [], "val_iou": [], "val_thr": []}

    ckpt_path = f"{CFG.CHECKPOINT_DIR}/levir_xnet_best.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=CFG.DEVICE,
                          weights_only=False)
        if ckpt.get("img_size", 256) == CFG.IMG_SIZE:
            model.load_state_dict(ckpt["state_dict"])
            best_f1        = ckpt.get("val_f1", 0.0)
            best_threshold = ckpt.get("threshold", CFG.THRESHOLD)
            start_epoch    = ckpt.get("epoch", 1) + 1
            for _ in range(start_epoch - 1):
                scheduler.step()
            print(f"  [RESUME] epoch {start_epoch}, F1={best_f1:.4f}")
        else:
            print("  [FRESH] IMG_SIZE changed, fresh start")
    else:
        print("  [FRESH] Starting from scratch")

    print("\n" + "=" * 65)
    print("  SNUNet v9 — Definitive Fix")
    print(f"  Device : {CFG.DEVICE}  |  IMG={CFG.IMG_SIZE}  "
          f"|  B={CFG.BATCH_SIZE}  |  Ep={start_epoch}→{CFG.EPOCHS}")
    print(f"  LR_HEAD={CFG.LR_HEAD}  LR_BACKBONE={CFG.LR_BACKBONE}")
    print("=" * 65)

    for epoch in range(start_epoch, CFG.EPOCHS + 1):

        # Stage transitions
        if epoch == CFG.STAGE2_START + 1 and current_stage < 2:
            print("\n>>> Stage 2: Unfreezing top 30% of ResNet50")
            model.unfreeze_top_backbone(0.30)
            optimizer = build_optimizer(model, stage=2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CFG.EPOCHS - epoch,
                eta_min=1e-7)
            current_stage = 2

        elif epoch == CFG.STAGE3_START + 1 and current_stage < 3:
            print("\n>>> Stage 3: Full fine-tuning")
            model.unfreeze_all()
            optimizer = build_optimizer(model, stage=3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CFG.EPOCHS - epoch,
                eta_min=1e-7)
            current_stage = 3

        tr_loss, tr_m = train_epoch(
            model, train_loader, optimizer, criterion, epoch)
        vl_loss, vl_m, val_thr = eval_epoch(
            model, val_loader, criterion)

        if epoch > 5:   # don't step during warmup
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        vram = (torch.cuda.memory_allocated() / 1e9
                if CFG.DEVICE.type == "cuda" else 0.0)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_f1"].append(vl_m["F1"])
        history["val_iou"].append(vl_m["IoU"])
        history["val_thr"].append(val_thr)

        print(f"Ep {epoch:3d}/{CFG.EPOCHS} S{current_stage} | "
              f"TrL={tr_loss:.4f} TrF1={tr_m['F1']:.4f} | "
              f"VlL={vl_loss:.4f} VlF1={vl_m['F1']:.4f} "
              f"IoU={vl_m['IoU']:.4f} "
              f"P={vl_m['Precision']:.4f} R={vl_m['Recall']:.4f} | "
              f"Thr={val_thr:.2f} LR={current_lr:.2e} "
              f"VRAM={vram:.1f}GB")

        if vl_m["F1"] > best_f1:
            best_f1        = vl_m["F1"]
            best_threshold = val_thr
            patience_c     = 0
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_f1":     vl_m["F1"],
                "val_iou":    vl_m["IoU"],
                "threshold":  best_threshold,
                "img_size":   CFG.IMG_SIZE,
            }, ckpt_path)
            print(f"  [BEST] F1={best_f1:.4f} "
                  f"IoU={vl_m['IoU']:.4f} Thr={best_threshold:.2f}")
        else:
            patience_c += 1
            if patience_c >= CFG.PATIENCE:
                print(f"\n[STOP] Early stopping ep {epoch}")
                break

    print(f"\n[DONE] Best Val F1={best_f1:.4f} "
          f"@ Thr={best_threshold:.2f}")
    return history