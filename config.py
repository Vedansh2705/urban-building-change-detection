"""
config.py - v8 WITH CHIPPED DATASET
──────────────────────────────────────
DATA_DIR now points to the pre-chipped 256x256 patches.
Run prep_chips.py once before training to create them.
"""
import os
import torch


class Config:

    SEED        = 42

    # IMPORTANT: points to chipped dataset, not original 1024x1024
    # Run prep_chips.py first to create this folder
    DATA_DIR    = r"C:\Users\ramsi\Desktop\LervirCD\LEVIR CD chipped"

    IMG_SIZE    = 256
    BATCH_SIZE  = 8
    GRAD_ACCUM  = 1
    NUM_WORKERS = 0

    MODEL_BASE           = 64
    USE_GRAD_CHECKPOINT  = True

    BACKBONE    = "resnet50_snunet"
    DROPOUT     = 0.1

    EPOCHS       = 200
    LR_HEAD      = 5e-4
    LR_BACKBONE  = 1e-5
    WEIGHT_DECAY = 1e-4
    PATIENCE     = 30

    POS_WEIGHT   = 3.0
    THRESHOLD    = 0.50

    STAGE2_START = 15
    STAGE3_START = 40

    CHECKPOINT_DIR = "./checkpoints"
    RESULTS_DIR    = "./results"

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")


CFG = Config()
os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CFG.RESULTS_DIR,    exist_ok=True)

if __name__ == "__main__":
    print("=" * 55)
    print("  Config v8 — Chipped Dataset")
    print("=" * 55)
    print(f"  Device      : {CFG.DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory/1e9
        print(f"  VRAM        : {vram:.1f} GB")
    print(f"  DATA_DIR    : {CFG.DATA_DIR}")
    exists = os.path.exists(CFG.DATA_DIR)
    print(f"  Exists      : {exists}")
    if not exists:
        print("  !! Run prep_chips.py first !!")
    print(f"  IMG_SIZE    : {CFG.IMG_SIZE}")
    print(f"  EPOCHS      : {CFG.EPOCHS}")
    print("=" * 55)