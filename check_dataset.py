"""
check_dataset.py
────────────────
FIXED VERSION - handles .DS_Store and nested path
"""
import os
from PIL import Image
import numpy as np

# ── FIXED Path (was double nested) ─���─────────────────────
DATA_DIR = r"C:\Users\ramsi\Desktop\levir_xnet_project\LEVIR-CD+\LEVIR-CD+"

# ── Only accept real image files (ignore .DS_Store etc) ──
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def get_images(folder):
    """Get only real image files, ignore hidden files like .DS_Store"""
    all_files = sorted(os.listdir(folder))
    images    = [f for f in all_files 
                 if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]
    hidden    = [f for f in all_files 
                 if f not in images]
    if hidden:
        print(f"  ⚠️  Ignored non-image files: {hidden}")
    return images


def check_split(split):
    print(f"\n── Checking [{split}] split ──────────────────────")
    
    a_dir     = os.path.join(DATA_DIR, split, "A")
    b_dir     = os.path.join(DATA_DIR, split, "B")
    label_dir = os.path.join(DATA_DIR, split, "label")

    # Check folders exist
    for d in [a_dir, b_dir, label_dir]:
        if os.path.exists(d):
            print(f"  ✅ Found : {d}")
        else:
            print(f"  ❌ MISSING: {d}")
            return

    # Count ONLY real images (skip .DS_Store etc)
    a_imgs     = get_images(a_dir)
    b_imgs     = get_images(b_dir)
    label_imgs = get_images(label_dir)

    print(f"\n  📁 A folder     : {len(a_imgs)} images")
    print(f"  📁 B folder     : {len(b_imgs)} images")
    print(f"  📁 label folder : {len(label_imgs)} images")

    # Check names match
    if a_imgs == b_imgs == label_imgs:
        print(f"  ✅ All filenames match perfectly!")
    else:
        print(f"  ⚠️  Filename mismatch!")
        # Show which files are extra
        a_set     = set(a_imgs)
        b_set     = set(b_imgs)
        label_set = set(label_imgs)
        
        extra_a = a_set - b_set - label_set
        extra_b = b_set - a_set - label_set
        extra_l = label_set - a_set - b_set
        
        if extra_a:
            print(f"  Extra in A     : {list(extra_a)[:5]}")
        if extra_b:
            print(f"  Extra in B     : {list(extra_b)[:5]}")
        if extra_l:
            print(f"  Extra in label : {list(extra_l)[:5]}")

    # Check first image details
    if a_imgs:
        sample_a     = Image.open(os.path.join(a_dir,     a_imgs[0]))
        sample_b     = Image.open(os.path.join(b_dir,     b_imgs[0]))
        sample_label = Image.open(os.path.join(label_dir, label_imgs[0]))

        print(f"\n  📷 Sample image name  : {a_imgs[0]}")
        print(f"  📐 T1 image size      : {sample_a.size}")
        print(f"  📐 T2 image size      : {sample_b.size}")
        print(f"  📐 Label size         : {sample_label.size}")
        print(f"  🎨 T1 image mode      : {sample_a.mode}")
        print(f"  🎨 Label mode         : {sample_label.mode}")

        # Check label values
        label_arr  = np.array(sample_label)
        unique_vals = np.unique(label_arr)
        change_pct  = (label_arr > 127).mean() * 100
        print(f"\n  🏷️  Label unique values : {unique_vals}")
        print(f"  📊 Changed pixels      : {change_pct:.2f}%")
        print(f"  📊 Unchanged pixels    : {100-change_pct:.2f}%")


# ── Run checks ────────────────────────────────────────────
print("=" * 55)
print("  LEVIR-CD+ Dataset Verification (FIXED)")
print("=" * 55)
print(f"\n  Dataset path: {DATA_DIR}")
print(f"  Path exists : {os.path.exists(DATA_DIR)}")

check_split("train")
check_split("test")

print("\n" + "=" * 55)
print("  ✅ Dataset check complete!")
print("=" * 55)