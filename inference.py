"""
inference.py — Run trained model on ANY two images
────────────────────────────────────────────────────
Usage examples:

  # Test a single pair of your own images:
  python inference.py --t1 my_before.png --t2 my_after.png

  # Test a specific chip from the test set:
  python inference.py --t1 "LEVIR CD chipped/test/A/test_001_00.png" \
                      --t2 "LEVIR CD chipped/test/B/test_001_00.png" \
                      --label "LEVIR CD chipped/test/label/test_001_00.png"

  # Test all chips from a specific original image:
  python inference.py --all-chips test_001

  # Test N random chips from the test set:
  python inference.py --random 12

Output saved to: ./results/inference_<name>.png
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG
from snunet_model import SNUNet_ECAM


# ── ImageNet normalisation — must match training ──────────────
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

transform = A.Compose([
    A.Resize(height=CFG.IMG_SIZE, width=CFG.IMG_SIZE),
    A.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ToTensorV2()
], additional_targets={"image2": "image"})


def denorm(tensor):
    """Convert normalised tensor back to displayable RGB image."""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


# ── Model loading ─────────────────────────────────────────────
def load_model():
    ckpt_path = f"{CFG.CHECKPOINT_DIR}/levir_xnet_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint found at {ckpt_path}")
        print("Train the model first with: python main.py")
        sys.exit(1)

    model = SNUNet_ECAM(
        base=getattr(CFG, 'MODEL_BASE', 64),
        use_checkpoint=False   # no checkpointing at inference
    ).to(CFG.DEVICE)

    ckpt = torch.load(ckpt_path, map_location=CFG.DEVICE,
                      weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    threshold  = ckpt.get("threshold", CFG.THRESHOLD)
    val_f1     = ckpt.get("val_f1", 0.0)
    epoch      = ckpt.get("epoch", "?")
    print(f"  [OK] Loaded checkpoint: epoch={epoch} "
          f"Val F1={val_f1:.4f} Threshold={threshold:.2f}")
    return model, threshold


# ── GradCAM ───────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model        = model
        self._gradients   = None
        self._activations = None
        self._hook_layer(model)

    def _hook_layer(self, model):
        enc = getattr(model, 'encoder', None)
        target = None
        if enc is not None:
            for child in reversed(list(enc.children())):
                if isinstance(child, torch.nn.Sequential):
                    blocks = list(child.children())
                    if blocks:
                        convs = [m for m in blocks[-1].modules()
                                 if isinstance(m, torch.nn.Conv2d)]
                        if convs:
                            target = convs[-1]
                            break
        if target is None:
            for m in model.modules():
                if isinstance(m, torch.nn.Conv2d):
                    target = m
        target.register_forward_hook(self._save_act)
        target.register_full_backward_hook(self._save_grad)
        print(f"  [GradCAM] Hooked: {type(target).__name__} "
              f"in_channels={target.in_channels}")

    def _save_act(self, module, inp, out):
        self._activations = out.detach()

    def _save_grad(self, module, gin, gout):
        self._gradients = gout[0].detach()

    def generate(self, img1_t, img2_t):
        self.model.zero_grad()
        cmap, _ = self.model(
            img1_t.to(CFG.DEVICE),
            img2_t.to(CFG.DEVICE))
        cmap.mean().backward()

        if self._gradients is None or self._activations is None:
            h = img1_t.shape[2]
            return np.zeros((h, h)), torch.sigmoid(cmap).squeeze().detach().cpu()

        weights = self._gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = torch.clamp(cam, min=0).squeeze().cpu().numpy()

        if cam.ndim == 0:
            cam = np.zeros((8, 8))

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        prob = torch.sigmoid(cmap).squeeze().detach().cpu()
        return cam, prob


def cam_overlay(img_np, cam, alpha=0.5):
    """Overlay GradCAM heatmap on image."""
    cam_r = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    hmap  = cv2.applyColorMap(
        (cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hmap  = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
    return (alpha * hmap + (1 - alpha) * (img_np * 255)).astype(np.uint8)


# ── Load one image pair ───────────────────────────────────────
def load_pair(t1_path, t2_path, label_path=None):
    img1 = np.array(Image.open(t1_path).convert("RGB"))
    img2 = np.array(Image.open(t2_path).convert("RGB"))

    # Handle images larger than 256px by centre-cropping
    h, w = img1.shape[:2]
    if h != CFG.IMG_SIZE or w != CFG.IMG_SIZE:
        print(f"  Image size {h}×{w} → resizing to {CFG.IMG_SIZE}×{CFG.IMG_SIZE}")

    mask = None
    if label_path and os.path.exists(label_path):
        mask_raw = np.array(Image.open(label_path).convert("L"))
        # Apply same resize as images
        result = transform(image=img1, image2=img2, mask=mask_raw)
        mask = (result["mask"].numpy() > 127).astype(np.uint8)
    else:
        result = transform(image=img1, image2=img2)

    img1_t = result["image"].unsqueeze(0)
    img2_t = result["image2"].unsqueeze(0)
    return img1_t, img2_t, mask


# ── Visualise one pair ────────────────────────────────────────
def visualise_pair(ax_row, img1_t, img2_t, mask, prob, cam,
                   threshold, title=""):
    """Fill one row of the figure: T1 | T2 | GT | Pred | GradCAM."""
    img1_np = denorm(img1_t.squeeze(0))
    img2_np = denorm(img2_t.squeeze(0))
    pred    = (prob.numpy() > threshold).astype(np.uint8)
    overlay = cam_overlay(img2_np, cam)

    ax_row[0].imshow(img1_np);            ax_row[0].set_title("T1 Before",  fontsize=9)
    ax_row[1].imshow(img2_np);            ax_row[1].set_title("T2 After",   fontsize=9)

    if mask is not None:
        ax_row[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    else:
        ax_row[2].imshow(np.zeros_like(pred), cmap="gray")
        ax_row[2].text(0.5, 0.5, "No label\nprovided",
                       ha="center", va="center",
                       transform=ax_row[2].transAxes,
                       fontsize=8, color="white")
    ax_row[2].set_title("GT Mask", fontsize=9)

    ax_row[3].imshow(pred, cmap="gray", vmin=0, vmax=1)
    ax_row[3].set_title("Predicted", fontsize=9)

    ax_row[4].imshow(overlay)
    ax_row[4].set_title("Grad-CAM", fontsize=9)

    if title:
        ax_row[0].set_ylabel(title, fontsize=8, rotation=90)

    # Compute quick metrics if label available
    if mask is not None:
        flat_p = pred.flatten()
        flat_g = mask.flatten()
        tp = ((flat_p==1) & (flat_g==1)).sum()
        fp = ((flat_p==1) & (flat_g==0)).sum()
        fn = ((flat_p==0) & (flat_g==1)).sum()
        prec = tp / (tp + fp + 1e-6)
        rec  = tp / (tp + fn + 1e-6)
        f1   = 2*prec*rec / (prec + rec + 1e-6)
        iou  = tp / (tp + fp + fn + 1e-6)
        ax_row[3].set_xlabel(
            f"F1={f1*100:.1f}%  IoU={iou*100:.1f}%",
            fontsize=7, color="green" if f1 > 0.8 else "orange")

    for ax in ax_row:
        ax.axis("off")


# ── Main routines ─────────────────────────────────────────────
def run_single(model, gradcam, threshold, t1, t2, label, save_path):
    """Visualise a single T1/T2 pair."""
    print(f"\n  T1: {t1}")
    print(f"  T2: {t2}")

    img1_t, img2_t, mask = load_pair(t1, t2, label)
    cam, prob = gradcam.generate(img1_t, img2_t)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    name = os.path.basename(t1)
    visualise_pair(axes, img1_t, img2_t, mask, prob, cam,
                   threshold, title=name)

    plt.suptitle(f"Change Detection + Grad-CAM  |  {name}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved → {save_path}")


def run_random(model, gradcam, threshold, n, save_path):
    """Visualise N random chips from the test set."""
    test_a = os.path.join(CFG.DATA_DIR, "test", "A")
    if not os.path.exists(test_a):
        print(f"ERROR: test/A not found at {test_a}")
        sys.exit(1)

    all_names = sorted([
        f for f in os.listdir(test_a)
        if f.lower().endswith((".png", ".jpg", ".tif"))
    ])
    chosen = random.sample(all_names, min(n, len(all_names)))
    print(f"\n  Visualising {len(chosen)} random test chips...")

    fig, axes = plt.subplots(len(chosen), 5,
                             figsize=(20, len(chosen) * 3.5))
    if len(chosen) == 1:
        axes = [axes]

    for i, name in enumerate(chosen):
        t1    = os.path.join(CFG.DATA_DIR, "test", "A",     name)
        t2    = os.path.join(CFG.DATA_DIR, "test", "B",     name)
        label = os.path.join(CFG.DATA_DIR, "test", "label", name)

        img1_t, img2_t, mask = load_pair(t1, t2, label)
        cam, prob = gradcam.generate(img1_t, img2_t)
        visualise_pair(axes[i], img1_t, img2_t, mask, prob, cam,
                       threshold, title=name)
        print(f"  [{i+1}/{len(chosen)}] {name}")

    plt.suptitle(f"Change Detection + Grad-CAM  |  {len(chosen)} random test chips",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved → {save_path}")


def run_all_chips(model, gradcam, threshold, stem, save_path):
    """Show all 16 chips from one original image (e.g. stem='test_001')."""
    test_a = os.path.join(CFG.DATA_DIR, "test", "A")
    chips  = sorted([
        f for f in os.listdir(test_a)
        if f.startswith(stem) and f.lower().endswith(".png")
    ])
    if not chips:
        print(f"ERROR: No chips found matching '{stem}' in {test_a}")
        sys.exit(1)

    print(f"\n  Found {len(chips)} chips for '{stem}'")
    fig, axes = plt.subplots(len(chips), 5,
                             figsize=(20, len(chips) * 3.0))
    if len(chips) == 1:
        axes = [axes]

    for i, name in enumerate(chips):
        t1    = os.path.join(CFG.DATA_DIR, "test", "A",     name)
        t2    = os.path.join(CFG.DATA_DIR, "test", "B",     name)
        label = os.path.join(CFG.DATA_DIR, "test", "label", name)

        img1_t, img2_t, mask = load_pair(t1, t2, label)
        cam, prob = gradcam.generate(img1_t, img2_t)
        visualise_pair(axes[i], img1_t, img2_t, mask, prob, cam,
                       threshold, title=name)
        print(f"  [{i+1}/{len(chips)}] {name}")

    plt.suptitle(f"All chips from '{stem}'  |  Change Detection + Grad-CAM",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved → {save_path}")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run change detection on custom images")
    parser.add_argument("--t1",         type=str, default=None,
        help="Path to T1 (Before) image")
    parser.add_argument("--t2",         type=str, default=None,
        help="Path to T2 (After) image")
    parser.add_argument("--label",      type=str, default=None,
        help="(Optional) Path to ground truth label image")
    parser.add_argument("--random",     type=int, default=None,
        metavar="N", help="Show N random test chips")
    parser.add_argument("--all-chips",  type=str, default=None,
        metavar="STEM", help="Show all chips from one image (e.g. test_001)")
    parser.add_argument("--out",        type=str, default=None,
        help="Output filename (default: auto-named in results/)")
    parser.add_argument("--threshold",  type=float, default=None,
        help="Override detection threshold (default: from checkpoint)")
    args = parser.parse_args()

    os.makedirs(CFG.RESULTS_DIR, exist_ok=True)

    print("=" * 55)
    print("  Change Detection Inference")
    print(f"  Device    : {CFG.DEVICE}")
    print(f"  DATA_DIR  : {CFG.DATA_DIR}")
    print("=" * 55)

    model, threshold = load_model()
    if args.threshold is not None:
        threshold = args.threshold
        print(f"  Threshold overridden to {threshold:.2f}")

    gradcam = GradCAM(model)

    # ── Mode selection ────────────────────────────────────────
    if args.t1 and args.t2:
        name      = os.path.splitext(os.path.basename(args.t1))[0]
        save_path = args.out or f"{CFG.RESULTS_DIR}/inference_{name}.png"
        run_single(model, gradcam, threshold,
                   args.t1, args.t2, args.label, save_path)

    elif args.random:
        save_path = args.out or f"{CFG.RESULTS_DIR}/inference_random{args.random}.png"
        run_random(model, gradcam, threshold, args.random, save_path)

    elif args.all_chips:
        save_path = args.out or f"{CFG.RESULTS_DIR}/inference_{args.all_chips}_allchips.png"
        run_all_chips(model, gradcam, threshold, args.all_chips, save_path)

    else:
        # Default: 6 random test chips if no arguments given
        print("\n  No arguments given — showing 6 random test chips.")
        print("  Run with --help to see all options.\n")
        save_path = args.out or f"{CFG.RESULTS_DIR}/inference_random6.png"
        run_random(model, gradcam, threshold, 6, save_path)

    print("\n" + "=" * 55)
    print("  Done!")
    print("=" * 55)