"""
gradcam_viz.py - FIXED
────────────────────────
Works with both LEVIRXNet (EfficientNet) and SNUNet.

Hook selection:
  LEVIRXNet → model.encoder.conv_head  (EfficientNet last conv)
  SNUNet    → model.conv3_0[-1]        (deepest encoder block)
"""
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from config import CFG


class GradCAM:
    def __init__(self, model):
        self.model        = model
        self._gradients   = None
        self._activations = None

        target = self._find_target_layer(model)
        print(f"  [GradCAM] Hooking layer: {type(target).__name__}")
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    @staticmethod
    def _find_target_layer(model):
        """
        Robustly find the last meaningful conv layer to hook.
        Works for ResNet50 (timm FeatureListNet), EfficientNet, and SNUNet.

        Priority:
          1. timm ResNet50 via features_only: last layer of layer4 (deepest features)
          2. EfficientNet conv_head (original code path)
          3. SNUNet conv3_0 deepest encoder block
          4. Generic fallback: last Conv2d in the entire model
        """
        enc = getattr(model, 'encoder', None)

        if enc is not None:
            # timm ResNet50 with features_only=True is a FeatureListNet.
            # Its stages are accessible as children in order:
            # 0=stem, 1=layer1, 2=layer2, 3=layer3, 4=layer4
            # We want the last BasicBlock/Bottleneck in layer4.
            children = list(enc.children())
            # Walk from deepest child backwards, find last Sequential with Conv2d
            for child in reversed(children):
                if isinstance(child, torch.nn.Sequential):
                    blocks = list(child.children())
                    if blocks:
                        last_block = blocks[-1]
                        # Return last conv inside the block
                        convs = [m for m in last_block.modules()
                                 if isinstance(m, torch.nn.Conv2d)]
                        if convs:
                            return convs[-1]

            # EfficientNet fallback
            if hasattr(enc, 'conv_head'):
                return enc.conv_head

        # SNUNet fallback
        if hasattr(model, 'conv3_0'):
            return model.conv3_0[-1]

        # Generic fallback: last Conv2d anywhere in the model
        last_conv = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            return last_conv

        raise RuntimeError("GradCAM: could not find any Conv2d layer to hook.")

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, gin, gout):
        self._gradients = gout[0].detach()

    def generate(self, img1, img2):
        """Generate Grad-CAM heatmap. Returns [0.0-1.0]"""
        self.model.eval()
        img1 = img1.to(CFG.DEVICE)
        img2 = img2.to(CFG.DEVICE)

        self.model.zero_grad()
        cmap, _ = self.model(img1, img2)
        cmap.mean().backward()

        weights = self._gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = torch.clamp(cam, min=0)
        cam = cam.squeeze().cpu().numpy()

        # Handle edge case where cam is 0-dim
        if cam.ndim == 0:
            cam = np.zeros((8, 8))

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay(self, img_np, cam, alpha=0.5):
        """Overlay heatmap on image. Red=high, Blue=low."""
        cam_r = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        hmap  = cv2.applyColorMap(
            (cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
        hmap  = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
        return (alpha * hmap + (1 - alpha) * img_np).astype(np.uint8)


def visualize_predictions(model, dataset, num_samples=6):
    """
    5-column grid:
    T1 Before | T2 After | GT Mask | Predicted | Grad-CAM
    """
    print(f"  Generating visualizations for {num_samples} samples...")

    gradcam = GradCAM(model)
    indices = random.sample(range(len(dataset)),
                            min(num_samples, len(dataset)))

    fig, axes = plt.subplots(num_samples, 5,
                             figsize=(20, num_samples * 3.5))

    titles = ["T1 (Before)", "T2 (After)",
              "GT Mask", "Predicted", "Grad-CAM"]
    for col, t in enumerate(titles):
        axes[0, col].set_title(t, fontsize=11,
                               fontweight="bold", pad=8)

    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])

    def denorm(tensor):
        img = tensor.permute(1, 2, 0).numpy()
        return np.clip(img * STD + MEAN, 0, 1)

    for row, idx in enumerate(indices):
        img1_t, img2_t, mask_t, _ = dataset[idx]
        img1_np = denorm(img1_t)
        img2_np = denorm(img2_t)

        with torch.no_grad():
            cmap, _ = model(
                img1_t.unsqueeze(0).to(CFG.DEVICE),
                img2_t.unsqueeze(0).to(CFG.DEVICE))
            pred = (torch.sigmoid(cmap) > 0.5
                    ).squeeze().cpu().numpy()

        try:
            cam = gradcam.generate(
                img1_t.unsqueeze(0),
                img2_t.unsqueeze(0))
            overlay = gradcam.overlay(
                (img2_np * 255).astype(np.uint8), cam)
        except Exception as e:
            overlay = (img2_np * 255).astype(np.uint8)
            print(f"  [WARN] GradCAM: {e}")

        axes[row, 0].imshow(img1_np)
        axes[row, 1].imshow(img2_np)
        axes[row, 2].imshow(mask_t.numpy(), cmap="gray",
                            vmin=0, vmax=1)
        axes[row, 3].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[row, 4].imshow(overlay)
        axes[row, 0].set_ylabel(f"Sample {row+1}",
                                fontsize=9, rotation=90)

        for col in range(5):
            axes[row, col].axis("off")

    plt.suptitle("Change Detection + Grad-CAM",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = f"{CFG.RESULTS_DIR}/predictions_gradcam.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Visualization saved -> {path}")


if __name__ == "__main__":
    print("  gradcam_viz.py OK!")