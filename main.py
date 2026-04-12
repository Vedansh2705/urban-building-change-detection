"""
main.py - v5
────────────────────────────────────────────────────────────────
Fixes over v4:
1. SNUNet_ECAM receives base + use_checkpoint from CFG
   (config is single source of truth)
2. os.makedirs called early — crash-proof before any save
3. persistent_workers=True + prefetch_factor to avoid worker
   respawn every epoch
4. drop_last=True on train loader — avoids partial-batch edge
   cases with gradient accumulation
5. torch.cuda.empty_cache() between training and test eval
6. Shows both total and trainable param counts after freeze
7. Graceful CFG attribute fallbacks (use getattr) so older
   config files don't break
────────────────────────────────────────────────────────────────
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from config       import CFG
from dataset      import LEVIRCDPlusDataset
from snunet_model import SNUNet_ECAM
from loss         import LEVIRXNetLoss
from train        import train
from evaluate     import (evaluate_test,
                           plot_confusion_matrix,
                           plot_training_history)
from gradcam_viz  import visualize_predictions


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic = True slows training slightly but ensures
    # fully reproducible results; disable if speed > repro
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_loader(dataset, *, shuffle: bool,
                drop_last: bool = False) -> DataLoader:
    """
    Centralised DataLoader factory so all loaders share the same
    worker / memory settings.

    persistent_workers keeps worker processes alive between epochs
    (no respawn overhead).  Only active when num_workers > 0.
    """
    num_workers = getattr(CFG, 'NUM_WORKERS', 4)
    persistent  = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )


def print_param_counts(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    frozen    = total - trainable
    print(f"  Total params     : {total:>12,}")
    print(f"  Trainable params : {trainable:>12,}  "
          f"({trainable / total * 100:.1f}%)")
    print(f"  Frozen params    : {frozen:>12,}  "
          f"({frozen / total * 100:.1f}%)")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    set_seed(CFG.SEED)

    # FIX 2 — create output dirs before anything else so we never
    # crash mid-run because a directory doesn't exist
    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CFG.RESULTS_DIR,    exist_ok=True)

    print("=" * 65)
    print("  SNUNet v5: Urban Building Change Detection")
    print("  Dataset  : LEVIR-CD+")
    print("  Model    : SNUNet-ECAM + ResNet50")
    print(f"  Device   : {CFG.DEVICE}")
    print(f"  IMG_SIZE : {CFG.IMG_SIZE}")
    print(f"  Batch    : {CFG.BATCH_SIZE}"
          f" × {getattr(CFG, 'GRAD_ACCUM', 1)}"
          f" = {CFG.BATCH_SIZE * getattr(CFG, 'GRAD_ACCUM', 1)}"
          f" effective")
    print("=" * 65)

    # ── 1. Dataset ────────────────────────────────────────────
    print("\n[1/5] Loading LEVIR-CD+ dataset...")
    train_ds = LEVIRCDPlusDataset(split="train")
    val_ds   = LEVIRCDPlusDataset(split="val")
    test_ds  = LEVIRCDPlusDataset(split="test")

    # FIX 3+4 — persistent workers, drop_last on train
    train_loader = make_loader(train_ds, shuffle=True,  drop_last=True)
    val_loader   = make_loader(val_ds,   shuffle=False, drop_last=False)
    test_loader  = make_loader(test_ds,  shuffle=False, drop_last=False)

    print(f"  Train : {len(train_ds):,} pairs  "
          f"→ {len(train_loader)} batches")
    print(f"  Val   : {len(val_ds):,} pairs")
    print(f"  Test  : {len(test_ds):,} pairs")

    # ── 2. Model ──────────────────────────────────────────────
    print("\n[2/5] Building SNUNet-ECAM...")

    # FIX 1 — pull base + use_checkpoint from CFG so config is
    # the single source of truth; fall back gracefully if absent
    model_base       = getattr(CFG, 'MODEL_BASE',        64)
    use_ckpt         = getattr(CFG, 'USE_GRAD_CHECKPOINT', True)

    model = SNUNet_ECAM(
        base=model_base,
        use_checkpoint=use_ckpt,
    ).to(CFG.DEVICE)

    # Stage 1: freeze backbone so random TIAM weights don't
    # damage pretrained ResNet50 in the first epochs
    model.freeze_backbone()
    print_param_counts(model)

    # ── 3. Training ───────────────────────────────────────────
    print("\n[3/5] Training...")
    criterion = LEVIRXNetLoss()
    history   = train(model, train_loader, val_loader, criterion)

    # FIX 5 — clear fragmented GPU memory before test evaluation
    torch.cuda.empty_cache()

    # ── 4. Test evaluation ────────────────────────────────────
    print("\n[4/5] Evaluating on test set...")
    ckpt_path = f"{CFG.CHECKPOINT_DIR}/levir_xnet_best.pth"
    ckpt = torch.load(
        ckpt_path,
        map_location=CFG.DEVICE,
        weights_only=False)

    model.load_state_dict(ckpt["state_dict"])
    best_threshold = ckpt.get("threshold", CFG.THRESHOLD)

    print(f"  Loaded best model  →  "
          f"Val F1={ckpt['val_f1']:.4f}  "
          f"@ epoch {ckpt['epoch']}  "
          f"Thr={best_threshold:.2f}")

    use_tta = getattr(CFG, 'USE_TTA', True)
    metrics, all_logits, all_masks = evaluate_test(
        model, test_loader, criterion,
        threshold=best_threshold,
        use_tta=use_tta)

    # ── 5. Visualisations ─────────────────────────────────────
    print("\n[5/5] Saving visualizations...")
    plot_training_history(history)
    plot_confusion_matrix(all_logits, all_masks,
                          threshold=best_threshold)
    visualize_predictions(model, test_ds, num_samples=6)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  [OK] ALL DONE!")
    print(f"  F1-Score  : {metrics['F1']        * 100:.2f}%")
    print(f"  IoU       : {metrics['IoU']        * 100:.2f}%")
    print(f"  Precision : {metrics['Precision']  * 100:.2f}%")
    print(f"  Recall    : {metrics['Recall']     * 100:.2f}%")
    print(f"  OA        : {metrics['OA']         * 100:.2f}%")
    print(f"  Threshold : {best_threshold:.2f}")
    print(f"  Results   → {CFG.RESULTS_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()