# =====================================================
# utils/models.py
# All model definitions — import from here everywhere
#
# Teachers (candidates):
#   VGG_Scratch           — custom 4-block VGG, trained from scratch
#   VGG_Pretrained        — VGG16-BN fine-tuned from ImageNet
#   ResNet_Scratch        — custom lightweight ResNet, trained from scratch
#   ResNet18_Pretrained   — ResNet18 fine-tuned from ImageNet  ← selected teacher
#
# Students (2×2 controlled comparison):
#
#   Architecture │ Scratch               │ Pretrained
#   ─────────────┼───────────────────────┼──────────────────────
#   MobileNetV2  │ MobileNetV2_Scratch   │ MobileNetV2_Pretrained
#   MobileNetV3  │ MobileNetV3_Scratch   │ MobileNetV3_Pretrained ← selected student
#
#   Claim 1 — Architecture effect (fix initialization, vary architecture):
#     Scratch:    V2_Scratch    vs V3_Scratch
#     Pretrained: V2_Pretrained vs V3_Pretrained
#
#   Claim 2 — Initialization effect (fix architecture, vary initialization):
#     V2: V2_Scratch vs V2_Pretrained
#     V3: V3_Scratch vs V3_Pretrained
#
#   Expected winner: MobileNetV3_Pretrained on both dimensions.
# =====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    vgg16_bn,           VGG16_BN_Weights,
    resnet18,           ResNet18_Weights,
    mobilenet_v2,       MobileNet_V2_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
)


# ══════════════════════════════════════════════════════
# TEACHER CANDIDATES
# ══════════════════════════════════════════════════════

# ── 1. VGG-Style (from scratch) ───────────────────────

class VGG_Scratch(nn.Module):
    """
    Custom 4-block VGG-style CNN trained from scratch on VWW.
    Input: 96×96 RGB. ~2.9M parameters.
    Included for comparison against pretrained VGG16-BN.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 96→48
            nn.Conv2d(3,   32,  3, padding=1, bias=False), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32,  32,  3, padding=1, bias=False), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 48→24
            nn.Conv2d(32,  64,  3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,  64,  3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 24→12
            nn.Conv2d(64,  128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: 12→6
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 128),          nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def forward(self, x):
        return self.classifier(self.features(x))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)


# ── 2. VGG16-BN (pretrained ImageNet) ────────────────

class VGG_Pretrained(nn.Module):
    """
    VGG16-BN with ImageNet weights, custom head for VWW.
    ~138M parameters. Progressive unfreeze: top at epoch 10, all at epoch 20.
    Included for comparison against ResNet18 pretrained.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        base = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        self.features   = base.features
        self.avgpool    = base.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(25088, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512,   128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x.view(x.size(0), -1))

    def unfreeze_top(self):
        for p in self.features[24:].parameters():
            p.requires_grad = True
        print("🔥 VGG16-BN: unfroze features[24:]")

    def unfreeze_all(self):
        for p in self.features.parameters():
            p.requires_grad = True
        print("🔥 VGG16-BN: unfroze all features")


# ── 3. ResNet (from scratch) ──────────────────────────

class _BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential() if (stride == 1 and in_ch == out_ch) else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class ResNet_Scratch(nn.Module):
    """
    Custom lightweight ResNet trained from scratch on VWW.
    Input: 96×96 RGB. ~0.7M parameters.
    Included for comparison against ResNet18 pretrained.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self._in  = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.layer1 = self._make(32,  2, stride=1)
        self.layer2 = self._make(64,  2, stride=2)
        self.layer3 = self._make(128, 2, stride=2)
        self.layer4 = self._make(256, 2, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _make(self, out_ch, n, stride):
        layers = [_BasicBlock(self._in, out_ch, stride)]
        self._in = out_ch
        for _ in range(1, n):
            layers.append(_BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.classifier(self.pool(x))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)


# ── 4. ResNet18 (pretrained ImageNet) ─────────────────

class ResNet18_Pretrained(nn.Module):
    """
    ResNet18 with ImageNet weights, custom head for VWW.
    ~11.2M parameters. SELECTED TEACHER.
    Progressive unfreeze: layer4 at epoch 10, full at epoch 20.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B, 512, 1, 1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.fc(self.backbone(x))

    def unfreeze_layer4(self):
        for name, p in self.backbone.named_parameters():
            if "7" in name:   # layer4 is child index 7 in ResNet18
                p.requires_grad = True
        print("🔥 ResNet18: unfroze layer4")

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("🔥 ResNet18: unfroze all layers")


# ══════════════════════════════════════════════════════
# STUDENT CANDIDATES — 2×2 controlled design
#
# Both architectures trained under both initialization
# conditions so architecture and initialization effects
# can be separated cleanly.
# ══════════════════════════════════════════════════════

# ── Shared inverted residual block (used by V2 scratch) ──

class _InvRes(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


# ── 5. MobileNetV2 — Scratch ──────────────────────────

class MobileNetV2_Scratch(nn.Module):
    """
    Custom MobileNetV2-inspired architecture trained from random init on VWW.
    Input: 96×96 RGB. ~0.8M parameters.

    NOTE: This is a truncated custom implementation, NOT the full MobileNetV2 spec.
    Compared against MobileNetV2_Pretrained to isolate the initialization effect.
    Compared against MobileNetV3_Scratch to isolate the architecture effect.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True)
        )
        self.features = nn.Sequential(
            _InvRes(32, 16, 1, 1),
            _InvRes(16, 24, 2, 6), _InvRes(24, 24, 1, 6),
            _InvRes(24, 32, 2, 6), _InvRes(32, 32, 1, 6),
            _InvRes(32, 64, 2, 6), _InvRes(64, 64, 1, 6),
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(512, num_classes)
        self._init_weights()

    def forward(self, x):
        x = self.initial(x)
        x = self.features(x)
        x = self.head(x)
        return self.classifier(x.view(x.size(0), -1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)


# ── 6. MobileNetV2 — Pretrained ───────────────────────

class MobileNetV2_Pretrained(nn.Module):
    """
    Standard MobileNetV2 with ImageNet weights, head replaced for VWW.
    ~3.4M parameters.

    Compared against MobileNetV2_Scratch  → isolates initialization effect on V2.
    Compared against MobileNetV3_Pretrained → isolates architecture effect (both pretrained).
    """
    def __init__(self, num_classes=2):
        super().__init__()
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_features, num_classes)
        self.model = base
        # Freeze backbone initially
        for name, p in self.model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def unfreeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = True
        print("🔥 MobileNetV2 pretrained: unfroze all")


# ── 7. MobileNetV3-Small — Scratch ────────────────────

class MobileNetV3_Scratch(nn.Module):
    """
    MobileNetV3-Small architecture trained from random initialization on VWW.
    ~2.5M parameters. Same architecture as MobileNetV3_Pretrained, weights=None.

    Compared against MobileNetV3_Pretrained → isolates initialization effect on V3.
    Compared against MobileNetV2_Scratch    → isolates architecture effect (both scratch).

    Expected to underperform MobileNetV3_Pretrained — confirming that pretrained
    weights provide meaningful benefit even on a binary classification task.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        base = mobilenet_v3_small(weights=None)   # <-- no pretrained weights
        in_features = base.classifier[3].in_features
        base.classifier[3] = nn.Linear(in_features, num_classes)
        self.model = base
        self._init_weights()

    def forward(self, x):
        return self.model(x)

    def _init_weights(self):
        """Kaiming init for all conv layers; normal init for linear layers."""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)


# ── 8. MobileNetV3-Small — Pretrained ─────────────────

class MobileNetV3_Pretrained(nn.Module):
    """
    MobileNetV3-Small with ImageNet weights, head replaced for VWW.
    ~2.5M parameters. SELECTED STUDENT for KD.

    Advantages over alternatives:
    - Hardware-Aware NAS: designed for mobile/edge latency targets
    - SE (Squeeze-and-Excitation) blocks: channel attention, better acc/param ratio
    - h-swish activations: more efficient than ReLU6 on hardware with lookup tables
    - Pretrained weights: strong initialization on only 7,000 training images
    - Confirmed winner by 2×2 comparison in notebook 10

    Progressive unfreeze: full network at epoch 10 (lightweight backbone, safe to unfreeze early).
    """
    def __init__(self, num_classes=2):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = base.classifier[3].in_features
        base.classifier[3] = nn.Linear(in_features, num_classes)
        self.model = base
        # Freeze backbone initially
        for name, p in self.model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def unfreeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = True
        print("🔥 MobileNetV3-Small pretrained: unfroze all")


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model, path="/tmp/_size_check.pth"):
    import os
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1e6
    os.remove(path)
    return size


# ── Registries ────────────────────────────────────────

TEACHER_REGISTRY = {
    "VGG (scratch)":         VGG_Scratch,
    "VGG16-BN (pretrained)": VGG_Pretrained,
    "ResNet (scratch)":      ResNet_Scratch,
    "ResNet18 (pretrained)": ResNet18_Pretrained,
}

STUDENT_REGISTRY = {
    # 2×2 grid — rows = architecture, cols = initialization
    "MobileNetV2 (scratch)":         MobileNetV2_Scratch,
    "MobileNetV2 (pretrained)":      MobileNetV2_Pretrained,
    "MobileNetV3-Small (scratch)":   MobileNetV3_Scratch,
    "MobileNetV3-Small (pretrained)":MobileNetV3_Pretrained,
}

# Combined registry for final results notebook
MODEL_REGISTRY = {**TEACHER_REGISTRY, **STUDENT_REGISTRY}
