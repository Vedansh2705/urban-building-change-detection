"""
dataset.py - v8 CROP NOT RESIZE
─────────────────────────────────
THE KEY FIX: RandomCrop instead of Resize for training.
Val/Test use CenterCrop.

This means the model sees full-resolution 256x256 patches,
not squished 1024->256 blobs.

Run prep_chips.py ONCE first to create the chipped dataset.
Then point DATA_DIR in config.py to the chipped folder.
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import CFG

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def get_image_names(folder):
    all_files = sorted(os.listdir(folder))
    return [f for f in all_files
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]


def get_transforms(split: str):
    """
    Training: chips are already 256x256 so just do flips.
    If loading original 1024x1024 images, RandomCrop extracts
    a 256x256 patch at full resolution.
    """
    if split == "train":
        return A.Compose([
            # If images are already chipped (256x256), Resize is a no-op.
            # If loading originals, this crops a random 256x256 patch
            # at FULL resolution — not squished.
            A.RandomCrop(height=CFG.IMG_SIZE, width=CFG.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={"image2": "image"})
    else:
        return A.Compose([
            # Center crop for deterministic val/test
            A.CenterCrop(height=CFG.IMG_SIZE, width=CFG.IMG_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={"image2": "image"})


class LEVIRCDPlusDataset(Dataset):
    def __init__(self, split: str = "train"):
        self.split     = split
        self.transform = get_transforms(
            "train" if split == "train" else "val")

        if split == "train":
            self.base_dir = os.path.join(CFG.DATA_DIR, "train")
        elif split == "val":
            self.base_dir = os.path.join(CFG.DATA_DIR, "val")
        else:
            self.base_dir = os.path.join(CFG.DATA_DIR, "test")

        src_dir = os.path.join(self.base_dir, "A")
        self.img_names = get_image_names(src_dir)
        print(f"  [OK] {split:5s} : {len(self.img_names):4d} images  "
              f"({self.base_dir})")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]

        img1 = np.array(Image.open(
            os.path.join(self.base_dir, "A", name)
        ).convert("RGB"))
        img2 = np.array(Image.open(
            os.path.join(self.base_dir, "B", name)
        ).convert("RGB"))
        mask = np.array(Image.open(
            os.path.join(self.base_dir, "label", name)
        ).convert("L"))

        mask   = (mask > 127).astype(np.uint8)
        result = self.transform(image=img1, image2=img2, mask=mask)

        img1_t    = result["image"]
        img2_t    = result["image2"]
        mask_t    = result["mask"].long()
        intensity = torch.tensor(
            float(mask_t.float().mean()), dtype=torch.float32)

        return img1_t, img2_t, mask_t, intensity