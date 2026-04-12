"""
prep_chips.py — Run ONCE before training
──────────────────────────────────────────
Chips 1024x1024 LEVIR-CD images into 256x256 patches.
This is what ALL published papers do — NOT resize/squish.

Input:  LEVIR CD/train/A/*.png  (1024x1024)
Output: LEVIR CD chipped/train/A/*.png  (256x256, 16 chips per image)

Usage:
    python prep_chips.py

Time: ~2-3 minutes for full dataset.
After running, update DATA_DIR in config.py to point to the chipped folder.
"""
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────────
SRC_ROOT  = r"C:\Users\ramsi\Desktop\LervirCD\LEVIR CD"
DST_ROOT  = r"C:\Users\ramsi\Desktop\LervirCD\LEVIR CD chipped"
CHIP_SIZE = 256
STRIDE    = 256   # non-overlapping — matches all published papers
SPLITS    = ["train", "val", "test"]
FOLDERS   = ["A", "B", "label"]
# ────────────────────────────────────────────────────────────


def chip_image(img_np, chip_size, stride):
    """Yield (row, col, chip_array) for all chips."""
    H, W = img_np.shape[:2]
    for r in range(0, H - chip_size + 1, stride):
        for c in range(0, W - chip_size + 1, stride):
            yield r, c, img_np[r:r+chip_size, c:c+chip_size]


def process_split(split):
    print(f"\n── {split} ──────────────────────────────")

    # Get image names from folder A
    src_a = os.path.join(SRC_ROOT, split, "A")
    names = sorted([
        f for f in os.listdir(src_a)
        if f.lower().endswith((".png", ".jpg", ".tif", ".tiff"))
    ])
    print(f"  Found {len(names)} image pairs")

    # Create output dirs
    for folder in FOLDERS:
        os.makedirs(os.path.join(DST_ROOT, split, folder), exist_ok=True)

    total_chips = 0
    for name in tqdm(names, desc=f"  Chipping {split}"):
        stem = os.path.splitext(name)[0]

        # Load all three (A, B, label) for this pair
        imgs = {}
        for folder in FOLDERS:
            path = os.path.join(SRC_ROOT, split, folder, name)
            pil  = Image.open(path)
            imgs[folder] = np.array(pil)

        # Chip all three with same coordinates
        H, W = imgs["A"].shape[:2]
        chip_coords = [
            (r, c)
            for r in range(0, H - CHIP_SIZE + 1, STRIDE)
            for c in range(0, W - CHIP_SIZE + 1, STRIDE)
        ]

        for idx, (r, c) in enumerate(chip_coords):
            chip_name = f"{stem}_{idx:03d}.png"
            for folder in FOLDERS:
                chip = imgs[folder][r:r+CHIP_SIZE, c:c+CHIP_SIZE]
                dst  = os.path.join(DST_ROOT, split, folder, chip_name)
                Image.fromarray(chip).save(dst)

        total_chips += len(chip_coords)

    print(f"  Total chips: {total_chips}  "
          f"({total_chips // len(names)} per image)")
    return total_chips


if __name__ == "__main__":
    print("=" * 55)
    print("  LEVIR-CD Image Chipping")
    print(f"  Source : {SRC_ROOT}")
    print(f"  Dest   : {DST_ROOT}")
    print(f"  Size   : {CHIP_SIZE}x{CHIP_SIZE}  stride={STRIDE}")
    print("=" * 55)

    total = 0
    for split in SPLITS:
        src_check = os.path.join(SRC_ROOT, split, "A")
        if os.path.exists(src_check):
            total += process_split(split)
        else:
            print(f"  [SKIP] {split} not found at {src_check}")

    print("\n" + "=" * 55)
    print(f"  Done! Total chips created: {total}")
    print(f"\n  NOW update config.py:")
    print(f'  DATA_DIR = r"{DST_ROOT}"')
    print("=" * 55)