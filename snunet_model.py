"""
snunet_model.py - v8 (OOM fix: spatial pooling + grad checkpointing)
──────────────────────────────────────────────────────────────────────
Root cause of OOM:
  TIAM S2 = (B, HW, HW) spatial attention.
  At B=32, scale-0 spatial=64×64 → HW=4096 → (32,4096,4096) = 32 GB.

Two complementary fixes applied here:

  1. Spatial pooling in TIAM.forward():
       When HW > max_spatial (default 1024 = 32×32), fa/fb are pooled
       to that smaller grid, full TIAM runs there, then the difference
       map is bilinearly upsampled back to original resolution.
       S2 is now at most (B, 1024, 1024) = 32 MB at B=8 vs 2 GB at B=32.

  2. Gradient checkpointing on TIAM calls (use_checkpoint=True in CFG):
       torch.utils.checkpoint recomputes TIAM activations during backward
       instead of storing them → ~60% activation memory saved, ~30% slower.
       Only active during training; eval/inference unchanged.

Together these allow B=8 + GRAD_ACCUM=4 (=32 effective) to fit in 24 GB.
Interface unchanged: forward() returns (change_map, change_inten).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import timm


# ─────────────────────────────────────────────────────────────────────
# TIAM — Temporospatial Interactive Attention Module
# ─────────────────────────────────────────────────────────────────────

class TIAM(nn.Module):
    """
    From CDNeXt (Wei et al., IJAEOG 2024).

    When input HW > max_spatial, features are pooled to a smaller grid,
    full TIAM attention is computed there (cheap), and the difference map
    is upsampled back. This caps S2 memory at O(B × max_spatial²).
    """

    def __init__(self, in_channels, max_spatial: int = 1024):
        super().__init__()
        self.max_spatial = max_spatial

        self.v_proj  = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1, bias=False))
        self.q1_proj = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1, bias=False))
        self.q2_proj = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1, bias=False))

        self.out1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels))
        self.out2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels))

    def _tiam_core(self, fa, fb):
        """Full TIAM at whatever resolution fa/fb arrive."""
        B, C, H, W = fa.shape
        HW = H * W

        v1 = self.v_proj(fa).view(B, C, HW)
        v2 = self.v_proj(fb).view(B, C, HW)
        q1 = self.q1_proj(fa).view(B, C, HW)
        q2 = self.q2_proj(fb).view(B, C, HW)
        k1 = q1.permute(0, 2, 1)   # (B, HW, C)
        k2 = q2.permute(0, 2, 1)

        # S1: temporal/channel similarity  (B, C, C)
        s1 = torch.softmax(
            torch.bmm(q1, k2) / (HW ** 0.5), dim=-1)

        # S2: spatial/perspective similarity  (B, HW, HW)
        # This is the expensive step — kept small via pooling in forward()
        s2 = torch.softmax(
            torch.bmm(k1, q2) / (C ** 0.5), dim=-1)

        # Rebuild fa: Z1 = S1ᵀ · V1 · S2
        z1 = torch.bmm(s1.permute(0, 2, 1), v1)   # (B, C, HW)
        z1 = torch.bmm(z1, s2)                     # (B, C, HW)
        fa_rebuilt = self.out1(z1.view(B, C, H, W)) + fa

        # Rebuild fb: Z2 = S1 · V2 · S2ᵀ
        z2 = torch.bmm(s1, v2)                         # (B, C, HW)
        z2 = torch.bmm(z2, s2.permute(0, 2, 1))        # (B, C, HW)
        fb_rebuilt = self.out2(z2.view(B, C, H, W)) + fb

        return torch.abs(fa_rebuilt - fb_rebuilt)

    def forward(self, fa, fb):
        B, C, H, W = fa.shape
        HW = H * W

        if HW > self.max_spatial:
            # Pool to a smaller spatial grid, run TIAM, upsample back
            stride = max(1, int((HW / self.max_spatial) ** 0.5))
            fa_p   = F.avg_pool2d(fa, stride, stride)
            fb_p   = F.avg_pool2d(fb, stride, stride)
            diff_p = self._tiam_core(fa_p, fb_p)
            return F.interpolate(diff_p, (H, W),
                                 mode='bilinear', align_corners=False)
        else:
            return self._tiam_core(fa, fb)


# ─────────────────────────────────────────────────────────────────────
# ECAM — Ensemble Channel Attention
# ─────────────────────────────────────────────────────────────────────

class ECAM(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = (self.avg_pool(x) + self.max_pool(x)).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1).expand_as(x)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def single_conv(in_ch, out_ch, dropout=0.0):
    layers = [
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────
# SNUNet_ECAM — main model
# ─────────────────────────────────────────────────────────────────────

class SNUNet_ECAM(nn.Module):
    """
    ResNet50 (Siamese) + TIAM (4 scales) + SNUNet Dense Decoder + ECAM.

    v8 vs v7:
      • use_checkpoint param added — wraps TIAM forward calls with
        torch.utils.checkpoint during training (~60% activation memory).
      • TIAM uses max_spatial=1024 pooling → S2 ≤ (B, 1024, 1024).
      • base default changed to 64 to match main.py CFG.MODEL_BASE=64.
    """

    def __init__(self, in_ch=3, num_classes=1,
                 base=64, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Siamese encoder — shared ResNet50
        self.encoder = timm.create_model(
            'resnet50', pretrained=True,
            features_only=True, out_indices=(0, 1, 2, 3))

        enc_chs = [64, 256, 512, 1024]   # timm ResNet50 outputs

        # Project to uniform base channels
        self.proj0 = nn.Sequential(
            nn.Conv2d(enc_chs[0], base,   1, bias=False),
            nn.BatchNorm2d(base),   nn.ReLU(inplace=True))
        self.proj1 = nn.Sequential(
            nn.Conv2d(enc_chs[1], base*2, 1, bias=False),
            nn.BatchNorm2d(base*2), nn.ReLU(inplace=True))
        self.proj2 = nn.Sequential(
            nn.Conv2d(enc_chs[2], base*4, 1, bias=False),
            nn.BatchNorm2d(base*4), nn.ReLU(inplace=True))
        self.proj3 = nn.Sequential(
            nn.Conv2d(enc_chs[3], base*8, 1, bias=False),
            nn.BatchNorm2d(base*8), nn.ReLU(inplace=True))

        nb = [base, base*2, base*4, base*8]

        # TIAM at each scale — max_spatial caps S2 to (B, ≤1024, ≤1024)
        self.tiam0 = TIAM(nb[0], max_spatial=1024)
        self.tiam1 = TIAM(nb[1], max_spatial=1024)
        self.tiam2 = TIAM(nb[2], max_spatial=1024)
        self.tiam3 = TIAM(nb[3], max_spatial=1024)

        # SNUNet Dense Nested Decoder
        self.conv0_1 = single_conv(nb[0] + nb[1],   nb[0], dropout=0.15)
        self.conv1_1 = single_conv(nb[1] + nb[2],   nb[1], dropout=0.15)
        self.conv2_1 = single_conv(nb[2] + nb[3],   nb[2], dropout=0.15)
        self.conv0_2 = single_conv(nb[0]*2 + nb[1], nb[0], dropout=0.15)
        self.conv1_2 = single_conv(nb[1]*2 + nb[2], nb[1], dropout=0.15)
        self.conv0_3 = single_conv(nb[0]*3 + nb[1], nb[0], dropout=0.15)

        self.up    = nn.Upsample(scale_factor=2, mode='bilinear',
                                 align_corners=True)
        self.ecam  = ECAM(nb[0] * 4)
        self.final = nn.Conv2d(nb[0] * 4, num_classes, 1)
        self.reg_head = nn.Sequential(
            nn.Conv2d(nb[0] * 4, nb[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb[0], 1, 1),
            nn.Sigmoid())

        self._init_new_weights()

    def _init_new_weights(self):
        for name, m in self.named_modules():
            if 'encoder' in name:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # CRITICAL: Zero-init TIAM output projections.
        # Kaiming init on out1/out2 means TIAM generates random attention
        # from epoch 1, poisoning all gradients. With zero init, TIAM starts
        # as a plain |fa - fb| difference and gradually learns attention.
        # This is the "Zero Init Residual" trick used in ResNet and ViT papers.
        for tiam in [self.tiam0, self.tiam1, self.tiam2, self.tiam3]:
            for proj in [tiam.out1, tiam.out2]:
                for m in proj.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.zeros_(m.weight)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.zeros_(m.weight)   # scale=0 not 1
                        nn.init.zeros_(m.bias)

    def _encode(self, x):
        f = self.encoder(x)
        return (self.proj0(f[0]), self.proj1(f[1]),
                self.proj2(f[2]), self.proj3(f[3]))

    def _tiam_call(self, tiam, fa, fb):
        """
        Dispatch TIAM with optional gradient checkpointing.
        Checkpointing saves ~60% activation memory at ~30% FLOP cost.
        use_reentrant=False avoids deprecation warning in PyTorch ≥ 2.0.
        """
        if self.use_checkpoint and self.training:
            return cp.checkpoint(tiam, fa, fb, use_reentrant=False)
        return tiam(fa, fb)

    def forward(self, img1, img2):
        f0A, f1A, f2A, f3A = self._encode(img1)
        f0B, f1B, f2B, f3B = self._encode(img2)

        # Attention-aware temporal difference at each scale
        d0 = self._tiam_call(self.tiam0, f0A, f0B)
        d1 = self._tiam_call(self.tiam1, f1A, f1B)
        d2 = self._tiam_call(self.tiam2, f2A, f2B)
        d3 = self._tiam_call(self.tiam3, f3A, f3B)

        # Dense nested decoder
        x0_1 = self.conv0_1(torch.cat([d0, self.up(d1)], 1))
        x1_1 = self.conv1_1(torch.cat([d1, self.up(d2)], 1))
        x2_1 = self.conv2_1(torch.cat([d2, self.up(d3)], 1))
        x0_2 = self.conv0_2(torch.cat([d0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([d1, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([d0, x0_1, x0_2, self.up(x1_2)], 1))

        H, W = img1.shape[2], img1.shape[3]
        kw   = dict(mode='bilinear', align_corners=False)
        cat_out = torch.cat([
            F.interpolate(d0,   (H, W), **kw),
            F.interpolate(x0_1, (H, W), **kw),
            F.interpolate(x0_2, (H, W), **kw),
            F.interpolate(x0_3, (H, W), **kw),
        ], dim=1)

        cat_out      = self.ecam(cat_out)
        change_map   = self.final(cat_out).squeeze(1)
        change_inten = self.reg_head(cat_out).squeeze(1)
        return change_map, change_inten

    # Stage control
    def freeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        print("  Stage 1: ResNet50 frozen, training decoder+TIAM only")

    def unfreeze_top_backbone(self, fraction=0.3):
        params = list(self.encoder.parameters())
        n = int(len(params) * fraction)
        for p in params[-n:]:
            p.requires_grad = True
        print(f"  Stage 2: Top {fraction*100:.0f}% ResNet50 unfrozen")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        print("  Stage 3: Full network unfrozen")


# ─────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = SNUNet_ECAM(base=64, use_checkpoint=True)
    model.freeze_backbone()
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    cmap, cinten = model(x1, x2)
    total  = sum(p.numel() for p in model.parameters())
    trainp = sum(p.numel() for p in model.parameters()
                 if p.requires_grad)
    print(f"  change_map   : {cmap.shape}")
    print(f"  change_inten : {cinten.shape}")
    print(f"  Total params : {total:,}")
    print(f"  Trainable    : {trainp:,}")
    print("  SNUNet v8 OK!")