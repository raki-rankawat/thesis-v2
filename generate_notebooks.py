"""
generate_notebooks.py
Generates all 13 thesis notebooks as valid Jupyter JSON.
Run: python3 generate_notebooks.py
"""

import json
from pathlib import Path

OUT = Path("/home/claude/thesis_v2")
OUT.mkdir(exist_ok=True)

# ── Helpers ───────────────────────────────────────────

def nb(cells):
    def make(ct, src):
        c = {
            "cell_type": ct,
            "metadata":  {},
            "source":    src.splitlines(keepends=True) if isinstance(src, str) else src,
        }
        if ct == "code":
            c["execution_count"] = None
            c["outputs"] = []
        return c
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": [make(ct, src) for ct, src in cells],
    }

def save(name, notebook):
    (OUT / name).write_text(json.dumps(notebook, indent=1))
    print(f"  ✅  {name}")


# ── Common setup cells (identical in every notebook) ──

SETUP = [
("code", """\
# ── Mount Drive & load utils ────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import sys, shutil, os
UTILS_SRC = "/content/drive/My Drive/thesis/utils"
if os.path.exists(UTILS_SRC):
    shutil.copytree(UTILS_SRC, "/content/utils", dirs_exist_ok=True)
    sys.path.insert(0, "/content")
    print("✅ utils loaded from Drive")
else:
    sys.path.insert(0, "/content")
    print("⚠️  Place the utils/ folder at: My Drive/thesis/utils/")
"""),
("code", """\
# ── Standard imports ────────────────────────────────────────────────
import os, time, random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utils.dataset import prepare_dataset, get_loaders, get_test_loader
from utils.models  import (
    VGG_Scratch, VGG_Pretrained,
    ResNet_Scratch, ResNet18_Pretrained,
    MobileNetV2_Scratch, MobileNetV3_Pretrained,
    count_params, model_size_mb, MODEL_REGISTRY,
)
from utils.train import (
    setup_device, set_seed, evaluate,
    train_model, train_model_three_phase,
    train_multi_seed, train_kd, plot_history,
)

device = setup_device(seed=41)
"""),
("code", """\
# ── Dataset setup ───────────────────────────────────────────────────
prepare_dataset()
"""),
]

# Standard hyperparameter reference (markdown cell reused in all training notebooks)
HPARAM_MD = """\
## Standardized hyperparameters

All models use identical settings to ensure fair comparison:

| Parameter | Scratch models | Pretrained models |
|-----------|---------------|-------------------|
| Batch size | 64 | 64 |
| Optimizer | Adam | Adam |
| Weight decay | 1e-4 | 1e-4 |
| Label smoothing | 0.1 | 0.1 |
| Augmentation | standard | standard |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Patience | 10 | 10 |
| Seeds | [41, 52, 63] | [41, 52, 63] |
| Max epochs | 50 | 30 |
| LR | 1e-3 | 3e-4 → 1e-4 → 3e-5 (3-phase) |

Pretrained models use fewer max epochs because transfer learning converges faster.
The three-phase progressive unfreeze is only applicable to pretrained models.
"""

SAVE_DIR_CELL = ('code', 'SAVE_DIR = "/content/drive/My Drive/Colab Notebooks"\n')


# ══════════════════════════════════════════════════════
# PHASE 1: TEACHER CANDIDATES
# ══════════════════════════════════════════════════════

# ── 01 VGG Scratch ────────────────────────────────────
save("01_Teacher_VGG_Scratch.ipynb", nb([
    ("markdown", """\
# 01 · Teacher Candidate — VGG-Style (from scratch)

Custom 4-block VGG-style CNN trained entirely from random initialization on the VWW dataset.
Included as a baseline teacher candidate for comparison against the pretrained VGG16-BN.

**Hypothesis:** The pretrained variant will outperform this model due to richer feature
representations learned on ImageNet, despite the domain gap.
"""),
    *SETUP,
    ("markdown", HPARAM_MD),
    SAVE_DIR_CELL,
    ("code", """\
train_loader, val_loader = get_loaders(batch_size=64, augmentation="standard")
"""),
    ("code", """\
results, best = train_multi_seed(
    model_fn     = VGG_Scratch,
    train_loader = train_loader,
    val_loader   = val_loader,
    device       = device,
    seeds        = [41, 52, 63],
    save_dir     = SAVE_DIR,
    name_prefix  = "vgg_scratch",
    pretrained   = False,
    # Standard scratch hyperparameters
    epochs          = 50,
    lr              = 1e-3,
    weight_decay    = 1e-4,
    label_smoothing = 0.1,
    patience        = 10,
)
"""),
    ("code", """\
plot_history(best, title=f"VGG Scratch (seed {best['seed']})")

accs = [r["best_acc"] for r in results]
print(f"\\nVGG Scratch  |  Mean: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%  |  "
      f"Best: {best['best_acc']*100:.2f}% (seed {best['seed']})")
print(f"Best checkpoint: {best['save_path']}")
"""),
]))


# ── 02 VGG Pretrained ─────────────────────────────────
save("02_Teacher_VGG_Pretrained.ipynb", nb([
    ("markdown", """\
# 02 · Teacher Candidate — VGG16-BN (pretrained ImageNet)

VGG16 with Batch Normalization, fine-tuned from ImageNet weights using three-phase
progressive unfreezing.

**Three-phase protocol:**
- Phase 1 (epochs 1–9):  backbone frozen, train head only at lr=3e-4
- Phase 2 (epochs 10–19): unfreeze features[24:] at lr=1e-4
- Phase 3 (epochs 20–30): unfreeze all features at lr=3e-5

**Note:** VGG16-BN has ~138M parameters — substantially over-parameterized for a
7,000-image binary task. Results expected to be competitive but potentially unstable
compared to ResNet18 due to the parameter/data ratio.
"""),
    *SETUP,
    ("markdown", HPARAM_MD),
    SAVE_DIR_CELL,
    ("code", """\
train_loader, val_loader = get_loaders(batch_size=64, augmentation="standard")
"""),
    ("code", """\
results, best = train_multi_seed(
    model_fn     = VGG_Pretrained,
    train_loader = train_loader,
    val_loader   = val_loader,
    device       = device,
    seeds        = [41, 52, 63],
    save_dir     = SAVE_DIR,
    name_prefix  = "vgg_pretrained",
    pretrained   = True,
    # Standard pretrained hyperparameters
    epochs          = 30,
    lr_phase1       = 3e-4,
    lr_phase2       = 1e-4,
    lr_phase3       = 3e-5,
    phase2_epoch    = 10,
    phase3_epoch    = 20,
    weight_decay    = 1e-4,
    label_smoothing = 0.1,
    patience        = 10,
)
"""),
    ("code", """\
plot_history(best, title=f"VGG16-BN Pretrained (seed {best['seed']})")

accs = [r["best_acc"] for r in results]
print(f"\\nVGG16-BN  |  Mean: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%  |  "
      f"Best: {best['best_acc']*100:.2f}% (seed {best['seed']})")
print(f"Best checkpoint: {best['save_path']}")
"""),
]))


# ── 03 ResNet Scratch ─────────────────────────────────
save("03_Teacher_ResNet_Scratch.ipynb", nb([
    ("markdown", """\
# 03 · Teacher Candidate — ResNet (from scratch)

Custom lightweight ResNet (4 residual stages, BasicBlock) trained from random
initialization on VWW.

**Architecture:** stem → layer1(32) → layer2(64) → layer3(128) → layer4(256) → GAP → fc
~0.7M parameters. Residual connections provide better gradient flow than the VGG scratch
baseline, expected to outperform it despite fewer parameters.
"""),
    *SETUP,
    ("markdown", HPARAM_MD),
    SAVE_DIR_CELL,
    ("code", """\
train_loader, val_loader = get_loaders(batch_size=64, augmentation="standard")
"""),
    ("code", """\
results, best = train_multi_seed(
    model_fn     = ResNet_Scratch,
    train_loader = train_loader,
    val_loader   = val_loader,
    device       = device,
    seeds        = [41, 52, 63],
    save_dir     = SAVE_DIR,
    name_prefix  = "resnet_scratch",
    pretrained   = False,
    epochs          = 50,
    lr              = 1e-3,
    weight_decay    = 1e-4,
    label_smoothing = 0.1,
    patience        = 10,
)
"""),
    ("code", """\
plot_history(best, title=f"ResNet Scratch (seed {best['seed']})")

accs = [r["best_acc"] for r in results]
print(f"\\nResNet Scratch  |  Mean: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%  |  "
      f"Best: {best['best_acc']*100:.2f}% (seed {best['seed']})")
print(f"Best checkpoint: {best['save_path']}")
"""),
]))


# ── 04 ResNet18 Pretrained ────────────────────────────
save("04_Teacher_ResNet18_Pretrained.ipynb", nb([
    ("markdown", """\
# 04 · Teacher Candidate — ResNet18 (pretrained ImageNet)

ResNet18 fine-tuned from ImageNet weights using three-phase progressive unfreezing.
**This is the expected winner and selected teacher for knowledge distillation.**

**Justification for selection:**
- 11.2M parameters: sufficient capacity gap over the ~2.5M student (≈4.5× ratio)
- Residual connections generalize better than VGG on small datasets
- Pretrained features provide rich initialization: avoids overfitting on 7,000 images
- Faster convergence than VGG16-BN with less risk of instability

**Three-phase protocol:**
- Phase 1 (epochs 1–9):  backbone frozen, head only at lr=3e-4
- Phase 2 (epochs 10–19): unfreeze layer4 at lr=1e-4
- Phase 3 (epochs 20–30): unfreeze full backbone at lr=3e-5
"""),
    *SETUP,
    ("markdown", HPARAM_MD),
    SAVE_DIR_CELL,
    ("code", """\
train_loader, val_loader = get_loaders(batch_size=64, augmentation="standard")
"""),
    ("code", """\
results, best = train_multi_seed(
    model_fn     = ResNet18_Pretrained,
    train_loader = train_loader,
    val_loader   = val_loader,
    device       = device,
    seeds        = [41, 52, 63],
    save_dir     = SAVE_DIR,
    name_prefix  = "resnet18_pretrained",
    pretrained   = True,
    epochs          = 30,
    lr_phase1       = 3e-4,
    lr_phase2       = 1e-4,
    lr_phase3       = 3e-5,
    phase2_epoch    = 10,
    phase3_epoch    = 20,
    weight_decay    = 1e-4,
    label_smoothing = 0.1,
    patience        = 10,
)
"""),
    ("code", """\
plot_history(best, title=f"ResNet18 Pretrained (seed {best['seed']})")

accs = [r["best_acc"] for r in results]
print(f"\\nResNet18 Pretrained  |  Mean: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%  |  "
      f"Best: {best['best_acc']*100:.2f}% (seed {best['seed']})")
print(f"Best checkpoint: {best['save_path']}")
"""),
]))


# ── 05 Teacher Comparison ─────────────────────────────
save("05_Teacher_Comparison.ipynb", nb([
    ("markdown", """\
# 05 · Teacher Comparison

Loads the best checkpoint for each teacher candidate and compares:
accuracy, parameter count, model size, and inference latency.

**Expected outcome:** ResNet18 (pretrained) wins on accuracy with a favourable
parameters/accuracy tradeoff. VGG16-BN may match accuracy but at 12× the parameter cost.

⚠️  Update the checkpoint paths below after running notebooks 01–04.
"""),
    *SETUP,
    SAVE_DIR_CELL,
    ("code", """\
# ── Update these after running notebooks 01–04 ──────────────────────
# Replace seed_XX with your best seed number for each model
TEACHER_CKPTS = {
    "VGG (scratch)":         (VGG_Scratch,          f"{SAVE_DIR}/vgg_scratch_seed_XX.pth"),
    "VGG16-BN (pretrained)": (VGG_Pretrained,        f"{SAVE_DIR}/vgg_pretrained_seed_XX.pth"),
    "ResNet (scratch)":      (ResNet_Scratch,         f"{SAVE_DIR}/resnet_scratch_seed_XX.pth"),
    "ResNet18 (pretrained)": (ResNet18_Pretrained,    f"{SAVE_DIR}/resnet18_pretrained_seed_XX.pth"),
}
"""),
    ("code", """\
_, val_loader = get_loaders(batch_size=64)

def measure_latency(model, device, runs=200, warmup=20):
    model.eval()
    dummy = torch.randn(1, 3, 96, 96, device=device)
    with torch.no_grad():
        for _ in range(warmup): model(dummy)
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs): model(dummy)
        if device.type == "cuda": torch.cuda.synchronize()
    return (time.time() - t0) / runs * 1000   # ms

rows = []
for name, (fn, ckpt) in TEACHER_CKPTS.items():
    if not os.path.exists(ckpt):
        print(f"⚠️  Skipping {name} — checkpoint not found"); continue
    m = fn().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    acc  = evaluate(m, val_loader, device)
    tp,_ = count_params(m)
    size = model_size_mb(m)
    lat  = measure_latency(m, device)
    rows.append({"Model": name, "Val Acc (%)": round(acc*100, 2),
                 "Params (M)": round(tp/1e6, 2), "Size (MB)": round(size, 2),
                 "Latency (ms)": round(lat, 3)})
    print(f"{name:28s}  acc={acc*100:.2f}%  params={tp/1e6:.2f}M  "
          f"size={size:.2f}MB  lat={lat:.3f}ms")
"""),
    ("code", """\
import pandas as pd
df = pd.DataFrame(rows).set_index("Model")
print("\\n" + df.to_string())

winner = df["Val Acc (%)"].idxmax()
print(f"\\n🏆 Selected teacher: {winner}  ({df.loc[winner,'Val Acc (%)']:.2f}%)")
print("   → Set TEACHER_CKPT in notebooks 09 and 10")
"""),
    ("code", """\
# ── Comparison charts ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
colors = ["#4878D0", "#4878D0", "#E8735A", "#E8735A"]   # blue=scratch, orange=pretrained

for ax, col in zip(axes, ["Val Acc (%)", "Params (M)", "Latency (ms)"]):
    bars = df[col].plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title(col); ax.set_xticklabels(df.index, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.4)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x()+p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", fontsize=8)

plt.suptitle("Teacher Candidate Comparison  (blue=scratch, orange=pretrained)", y=1.02)
plt.tight_layout(); plt.show()
"""),
]))


# ══════════════════════════════════════════════════════
# PHASE 2: STUDENT CANDIDATES
# ══════════════════════════════════════════════════════

# ── 06 MobileNetV2 Scratch ────────────────────────────
save("06_Student_MobileNetV2_Scratch.ipynb", nb([
    ("markdown", """\
# 06 · Student Candidate — MobileNetV2 (from scratch)

Custom MobileNetV2-inspired architecture trained from random initialization.
This is the standalone student **baseline before any knowledge distillation**.

**Architecture:** initial conv → 7 inverted residual blocks → pointwise head → GAP → fc
~0.8M parameters. Depthwise-separable convolutions reduce computation vs standard convolutions.

Included to demonstrate the performance gap between scratch training and pretrained
initialization, and to show the effect of KD on both starting conditions.
"""),
    *SETUP,
    ("markdown", HPARAM_MD),
    SAVE_DIR_CELL,
    ("code", """\
train_loader, val_loader = get_loaders(batch_size=64, augmentation="standard")
"""),
    ("code", """\
results, best = train_multi_seed(
    model_fn     = MobileNetV2_Scratch,
    train_loader = train_loader,
    val_loader   = val_loader,
    device       = device,
    seeds        = [41, 52, 63],
    save_dir     = SAVE_DIR,
    name_prefix  = "mobilenetv2_baseline",
    pretrained   = False,
    epochs          = 50,
    lr              = 1e-3,
    weight_decay    = 1e-4,
    label_smoothing = 0.1,
    patience        = 10,
)
"""),
    ("code", """\
plot_history(best, title=f"MobileNetV2 Scratch (seed {best['seed']})")

accs = [r["best_acc"] for r in results]
print(f"\\nMobileNetV2 Scratch  |  Mean: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
print(f"Best: {best['best_acc']*100:.2f}% (seed {best['seed']})")
print(f"Checkpoint: {best['save_path']}")
"""),
]))


# ── 07 MobileNetV3 Pretrained ─────────────────────────
save("07_Student_MobileNetV3_Pretrained.ipynb", nb([
    ("markdown", """\
# 07 · Student Candidate — MobileNetV3-Small (pretrained ImageNet)

MobileNetV3-Small fine-tuned from ImageNet weights.
**This is the selected student for knowledge distillation.**

**Justification for selection over MobileNetV2-scratch:**
- Hardware-Aware NAS: designed explicitly for mobile/edge inference latency
- SE (Squeeze-and-Excitation) blocks: channel attention improves accuracy/param ratio
- h-swish activations: more efficient than ReLU6 on hardware with lookup tables
- Pretrained weights: better feature initialization on only 7,000 images
- ~2.5M parameters: suitable for STM32 deployment after INT8 quantization

**Two-phase protocol:**
- Phase 1 (epochs 1–9):  backbone frozen, classifier head only at lr=3e-4
- Phase 2 (epochs 10–25): full unfreeze at lr=1e-4
"""),
    *SETUP,
    ("markdown", HPARAM_MD),
    SAVE_DIR_CELL,
    ("code", """\
train_loader, val_loader = get_loaders(batch_size=64, augmentation="standard")
"""),
    ("code", """\
# MobileNetV3 uses two-phase (not three) — backbone is lighter, full unfreeze is safe at ep 10
results, best = train_multi_seed(
    model_fn     = MobileNetV3_Pretrained,
    train_loader = train_loader,
    val_loader   = val_loader,
    device       = device,
    seeds        = [41, 52, 63],
    save_dir     = SAVE_DIR,
    name_prefix  = "mobilenetv3_baseline",
    pretrained   = True,
    epochs          = 25,
    lr_phase1       = 3e-4,
    lr_phase2       = 1e-4,
    lr_phase3       = 1e-4,   # phase3 same as phase2 (effectively two-phase)
    phase2_epoch    = 10,
    phase3_epoch    = 999,    # disable phase3 transition
    weight_decay    = 1e-4,
    label_smoothing = 0.1,
    patience        = 10,
)
"""),
    ("code", """\
plot_history(best, title=f"MobileNetV3-Small Pretrained (seed {best['seed']})")

accs = [r["best_acc"] for r in results]
print(f"\\nMobileNetV3-Small  |  Mean: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
print(f"Best: {best['best_acc']*100:.2f}% (seed {best['seed']})")
print(f"Checkpoint: {best['save_path']}")
"""),
]))


# ── 08 Student Comparison ─────────────────────────────
save("08_Student_Comparison.ipynb", nb([
    ("markdown", """\
# 08 · Student Comparison

Head-to-head evaluation of both student candidates.
Selects the student model for knowledge distillation.

⚠️  Update checkpoint paths after running notebooks 06–07.
"""),
    *SETUP,
    SAVE_DIR_CELL,
    ("code", """\
# ── Update seed numbers after running 06–07 ─────────────────────────
STUDENT_CKPTS = {
    "MobileNetV2 (scratch)":          (MobileNetV2_Scratch,    f"{SAVE_DIR}/mobilenetv2_baseline_seed_XX.pth"),
    "MobileNetV3-Small (pretrained)": (MobileNetV3_Pretrained, f"{SAVE_DIR}/mobilenetv3_baseline_seed_XX.pth"),
}
"""),
    ("code", """\
_, val_loader = get_loaders(batch_size=64)

def measure_latency(model, device, runs=200, warmup=20):
    model.eval()
    dummy = torch.randn(1, 3, 96, 96, device=device)
    with torch.no_grad():
        for _ in range(warmup): model(dummy)
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs): model(dummy)
        if device.type == "cuda": torch.cuda.synchronize()
    return (time.time() - t0) / runs * 1000

rows = []
for name, (fn, ckpt) in STUDENT_CKPTS.items():
    if not os.path.exists(ckpt):
        print(f"⚠️  Skipping {name}"); continue
    m = fn().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    acc  = evaluate(m, val_loader, device)
    tp,_ = count_params(m)
    size = model_size_mb(m)
    lat  = measure_latency(m, device)
    rows.append({"Model": name, "Val Acc (%)": round(acc*100,2),
                 "Params (M)": round(tp/1e6,2), "Size (MB)": round(size,2),
                 "Latency (ms)": round(lat,3)})
    print(f"{name:42s}  {acc*100:.2f}%  {tp/1e6:.2f}M  {size:.2f}MB  {lat:.3f}ms")
"""),
    ("code", """\
import pandas as pd
df = pd.DataFrame(rows).set_index("Model")
print("\\n" + df.to_string())
winner = df["Val Acc (%)"].idxmax()
print(f"\\n🏆 Selected student: {winner}  ({df.loc[winner,'Val Acc (%)']:.2f}%)")
print("   → Set STUDENT_MODEL_FN and STUDENT_CKPT in notebooks 09 and 10")
"""),
    ("code", """\
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
colors = ["#4878D0", "#55A868"]
for ax, col in zip(axes, ["Val Acc (%)", "Params (M)", "Latency (ms)"]):
    df[col].plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title(col); ax.set_xticklabels(df.index, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.4)
plt.suptitle("Student Candidate Comparison", y=1.02)
plt.tight_layout(); plt.show()
"""),
]))


# ══════════════════════════════════════════════════════
# PHASE 3: KNOWLEDGE DISTILLATION
# ══════════════════════════════════════════════════════

# ── 09 KD Ablation ────────────────────────────────────
save("09_KD_Ablation.ipynb", nb([
    ("markdown", """\
# 09 · Knowledge Distillation — Hyperparameter Ablation

Sensitivity analysis for KD hyperparameters: temperature T and weighting alpha.

**Fixed:** best teacher (ResNet18 pretrained), best student (MobileNetV3), KD-FT mode, seed=41.
**Varied:** T ∈ {2, 4, 8} and alpha ∈ {0.5, 0.7, 0.9}.

**Theoretical justification (Hinton et al., 2015):**
- Temperature T controls the softness of the teacher's output distribution.
  Higher T reveals more inter-class similarity information (dark knowledge).
  T ∈ [2, 8] is recommended for standard classification tasks.
- Alpha balances hard labels (ground truth) vs soft labels (teacher distribution).
  Higher alpha → student relies more on teacher's soft targets.

This ablation justifies our final choice of T=4, alpha=0.7 used in notebook 10.
"""),
    *SETUP,
    SAVE_DIR_CELL,
    ("code", """\
# ── Set these after running notebooks 04 and 07 ──────────────────────
TEACHER_CKPT     = f"{SAVE_DIR}/resnet18_pretrained_seed_XX.pth"
STUDENT_CKPT     = f"{SAVE_DIR}/mobilenetv3_baseline_seed_XX.pth"
TEACHER_MODEL_FN = ResNet18_Pretrained
STUDENT_MODEL_FN = MobileNetV3_Pretrained
"""),
    ("code", """\
train_loader, val_loader = get_loaders(batch_size=64, augmentation="minimal")

# Load teacher once
teacher = TEACHER_MODEL_FN().to(device)
teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=device))
teacher_acc = evaluate(teacher, val_loader, device)
print(f"Teacher accuracy: {teacher_acc*100:.2f}%")

# Student baseline (pre-KD)
student_base = STUDENT_MODEL_FN().to(device)
student_base.load_state_dict(torch.load(STUDENT_CKPT, map_location=device))
baseline_acc = evaluate(student_base, val_loader, device)
print(f"Student baseline: {baseline_acc*100:.2f}%")
"""),
    ("code", """\
# ── Ablation grid ────────────────────────────────────────────────────
import copy

ABLATION_GRID = [
    {"T": 2.0, "alpha": 0.7, "label": "T=2, α=0.7"},
    {"T": 4.0, "alpha": 0.5, "label": "T=4, α=0.5"},
    {"T": 4.0, "alpha": 0.7, "label": "T=4, α=0.7 ← default"},
    {"T": 4.0, "alpha": 0.9, "label": "T=4, α=0.9"},
    {"T": 8.0, "alpha": 0.7, "label": "T=8, α=0.7"},
]

ablation_results = []

for cfg in ABLATION_GRID:
    print(f"\\n{'─'*50}")
    print(f"Running: {cfg['label']}")
    set_seed(41)
    student = STUDENT_MODEL_FN().to(device)
    student.load_state_dict(torch.load(STUDENT_CKPT, map_location=device))   # FT mode

    save_path = f"{SAVE_DIR}/ablation_{cfg['label'].replace(' ','_').replace('=','')}.pth"

    best_acc, elapsed = train_kd(
        student=student, teacher=teacher,
        train_loader=train_loader, val_loader=val_loader, device=device,
        epochs=20, lr=1e-3, weight_decay=1e-4,
        temperature=cfg["T"], alpha=cfg["alpha"],
        patience=8, save_path=save_path, verbose=False,
    )
    delta = best_acc - baseline_acc
    ablation_results.append({
        "Config": cfg["label"], "T": cfg["T"], "alpha": cfg["alpha"],
        "KD Val Acc (%)": round(best_acc*100, 2),
        "Δ vs baseline (%)": round(delta*100, 2),
        "Time (min)": round(elapsed, 1),
    })
    print(f"  → {best_acc*100:.2f}%  (Δ {delta*100:+.2f}%)")
"""),
    ("code", """\
import pandas as pd
df_abl = pd.DataFrame(ablation_results).set_index("Config")
print("\\n" + df_abl.to_string())

best_cfg = df_abl["KD Val Acc (%)"].idxmax()
print(f"\\n✅ Best config: {best_cfg}  ({df_abl.loc[best_cfg,'KD Val Acc (%)']:.2f}%)")
print(f"   Baseline: {baseline_acc*100:.2f}%  |  Teacher: {teacher_acc*100:.2f}%")
print("\\n→ Use best config in notebook 10_KD_Main.ipynb")
"""),
    ("code", """\
# ── Sensitivity plot ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(df_abl.index, df_abl["KD Val Acc (%)"], color="#4878D0", edgecolor="white")
ax.axhline(baseline_acc*100, color="gray",   linestyle="--", label=f"Student baseline ({baseline_acc*100:.2f}%)")
ax.axhline(teacher_acc*100,  color="#E8735A", linestyle="--", label=f"Teacher ({teacher_acc*100:.2f}%)")
ax.set_ylabel("Val Accuracy (%)"); ax.set_title("KD Hyperparameter Sensitivity")
ax.set_ylim(min(df_abl["KD Val Acc (%)"].min(), baseline_acc*100) - 2, 100)
for p in bars:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x()+p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=8)
ax.legend(); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.show()
"""),
]))


# ── 10 KD Main ────────────────────────────────────────
save("10_KD_Main.ipynb", nb([
    ("markdown", """\
# 10 · Knowledge Distillation — Main Experiment

Full KD experiment with selected teacher and student, across 3 seeds.
Compares two student initialization strategies:

- **KD-FT**: student initialized from pretrained baseline, then KD training
- **KD-Scratch**: student initialized randomly, then KD training

**Hyperparameters:** T=4.0, alpha=0.7 (justified by ablation in notebook 09).

**Expected outcome:** KD-FT outperforms KD-Scratch because the student already has
useful pretrained features before distillation begins.
"""),
    *SETUP,
    SAVE_DIR_CELL,
    ("code", """\
# ── Set after running notebooks 04, 07, and 09 ───────────────────────
TEACHER_CKPT     = f"{SAVE_DIR}/resnet18_pretrained_seed_XX.pth"
STUDENT_CKPT     = f"{SAVE_DIR}/mobilenetv3_baseline_seed_XX.pth"
TEACHER_MODEL_FN = ResNet18_Pretrained
STUDENT_MODEL_FN = MobileNetV3_Pretrained

KD_TEMPERATURE = 4.0
KD_ALPHA       = 0.7
KD_EPOCHS      = 30
KD_LR          = 1e-3
KD_SEEDS       = [41, 52, 63]
"""),
    ("code", """\
train_loader, val_loader = get_loaders(batch_size=64, augmentation="minimal")

teacher = TEACHER_MODEL_FN().to(device)
teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=device))
teacher_acc = evaluate(teacher, val_loader, device)
print(f"Teacher val accuracy: {teacher_acc*100:.2f}%")
"""),
    ("code", """\
# ── KD-FT: initialize student from pretrained baseline ───────────────
print("\\n" + "="*55 + "\\nKD-FT (fine-tune from pretrained baseline)\\n" + "="*55)
kd_ft_results = []

for seed in KD_SEEDS:
    print(f"\\nSeed {seed}")
    set_seed(seed)
    student = STUDENT_MODEL_FN().to(device)
    student.load_state_dict(torch.load(STUDENT_CKPT, map_location=device))
    baseline_acc = evaluate(student, val_loader, device)

    best_acc, elapsed = train_kd(
        student=student, teacher=teacher,
        train_loader=train_loader, val_loader=val_loader, device=device,
        epochs=KD_EPOCHS, lr=KD_LR, weight_decay=1e-4,
        temperature=KD_TEMPERATURE, alpha=KD_ALPHA,
        patience=10, save_path=f"{SAVE_DIR}/kd_ft_seed_{seed}.pth",
    )
    kd_ft_results.append({
        "seed": seed, "baseline": baseline_acc,
        "kd_acc": best_acc, "delta": best_acc - baseline_acc, "elapsed": elapsed
    })
"""),
    ("code", """\
# ── KD-Scratch: initialize student from random weights ───────────────
print("\\n" + "="*55 + "\\nKD-Scratch (random initialization)\\n" + "="*55)
kd_scratch_results = []

for seed in KD_SEEDS:
    print(f"\\nSeed {seed}")
    set_seed(seed)
    student = STUDENT_MODEL_FN().to(device)   # fresh random init — do NOT load baseline
    scratch_acc = evaluate(student, val_loader, device)
    print(f"  Random init accuracy: {scratch_acc*100:.2f}%")

    best_acc, elapsed = train_kd(
        student=student, teacher=teacher,
        train_loader=train_loader, val_loader=val_loader, device=device,
        epochs=KD_EPOCHS, lr=KD_LR, weight_decay=1e-4,
        temperature=KD_TEMPERATURE, alpha=KD_ALPHA,
        patience=10, save_path=f"{SAVE_DIR}/kd_scratch_seed_{seed}.pth",
    )
    kd_scratch_results.append({
        "seed": seed, "baseline": scratch_acc,
        "kd_acc": best_acc, "delta": best_acc - scratch_acc, "elapsed": elapsed
    })
"""),
    ("code", """\
# ── Summary ──────────────────────────────────────────────────────────
import pandas as pd

def summarize(results, label):
    accs = [r["kd_acc"] for r in results]
    return {
        "Condition": label,
        "Mean Acc (%)": round(np.mean(accs)*100, 2),
        "Std (%)":      round(np.std(accs)*100,  2),
        "Best Acc (%)": round(max(accs)*100,      2),
        "Δ vs baseline (%)": round((np.mean(accs) - np.mean([r["baseline"] for r in results]))*100, 2),
    }

student_base_acc = evaluate(
    STUDENT_MODEL_FN().to(device).__class__(
        **{}
    ), val_loader, device
) if False else None  # placeholder — filled from notebook 07 results

rows = [
    {"Condition": "Student baseline (no KD)",
     "Mean Acc (%)": "→ see nb 07", "Std (%)": "—", "Best Acc (%)": "—", "Δ vs baseline (%)": "—"},
    summarize(kd_ft_results,     "KD-FT"),
    summarize(kd_scratch_results,"KD-Scratch"),
    {"Condition": "Teacher (ResNet18)",
     "Mean Acc (%)": round(teacher_acc*100, 2), "Std (%)": "—", "Best Acc (%)": "—", "Δ vs baseline (%)": "—"},
]

df = pd.DataFrame(rows).set_index("Condition")
print("\\n" + df.to_string())
print("\\nBest KD-FT checkpoint:     ", f"{SAVE_DIR}/kd_ft_seed_{max(kd_ft_results, key=lambda r:r['kd_acc'])['seed']}.pth")
print("Best KD-Scratch checkpoint:", f"{SAVE_DIR}/kd_scratch_seed_{max(kd_scratch_results, key=lambda r:r['kd_acc'])['seed']}.pth")
"""),
]))


# ══════════════════════════════════════════════════════
# PHASE 4: COMPRESSION PIPELINE
# ══════════════════════════════════════════════════════

# ── 11 Pruning ────────────────────────────────────────
save("11_Pruning.ipynb", nb([
    ("markdown", """\
# 11 · Pruning (Additional Experiment)

Applies unstructured L1 magnitude pruning to the best KD-trained student
at multiple sparsity levels, followed by short fine-tuning.

## Important limitation

**Unstructured pruning does not reduce inference latency on STM32.**
The X-CUBE-AI runtime executes dense matrix operations — zero-valued weights
do not skip computation. Pruning reduces the number of non-zero parameters
and on-disk model size, but deployed inference time and RAM usage are unchanged.

This experiment demonstrates accuracy degradation as a function of sparsity,
and confirms that **quantization (notebook 12) is the more deployment-relevant
compression technique** for the STM32 target hardware.

Sparsity levels tested: 20%, 30%, 50%.
"""),
    *SETUP,
    SAVE_DIR_CELL,
    ("code", """\
import torch.nn.utils.prune as prune
import copy

train_loader, val_loader = get_loaders(batch_size=64, augmentation="standard")
"""),
    ("code", """\
# ── Point to best KD checkpoint ──────────────────────────────────────
PRUNE_CKPT  = f"{SAVE_DIR}/kd_ft_seed_XX.pth"
MODEL_FN    = MobileNetV3_Pretrained

base_model = MODEL_FN().to(device)
base_model.load_state_dict(torch.load(PRUNE_CKPT, map_location=device))
base_acc = evaluate(base_model, val_loader, device)
tp, _ = count_params(base_model)
print(f"Base model accuracy: {base_acc*100:.2f}%  |  Params: {tp/1e6:.2f}M")
"""),
    ("code", """\
# ── Pruning sweep ─────────────────────────────────────────────────────
from utils.train import train_epoch

FT_EPOCHS    = 5
FT_LR        = 1e-4
PRUNE_LEVELS = [0.20, 0.30, 0.50]

prune_results = []

for amount in PRUNE_LEVELS:
    print(f"\\n{'='*50}\\nSparsity target: {amount*100:.0f}%")

    model = copy.deepcopy(base_model)
    conv_layers = [(m, "weight") for m in model.modules() if isinstance(m, nn.Conv2d)]

    # Apply L1 unstructured pruning
    for layer, param in conv_layers:
        prune.l1_unstructured(layer, name=param, amount=amount)

    acc_pruned = evaluate(model, val_loader, device)
    print(f"  After pruning  (before FT): {acc_pruned*100:.2f}%")

    # Short fine-tuning
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=FT_LR)
    for ep in range(1, FT_EPOCHS + 1):
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate(model, val_loader, device)
        print(f"  FT {ep}/{FT_EPOCHS}  train={ta*100:.2f}%  val={va*100:.2f}%")

    # Make pruning permanent
    for layer, param in conv_layers:
        prune.remove(layer, param)

    acc_ft = evaluate(model, val_loader, device)

    # Measure actual sparsity
    zeros = sum((p == 0).sum().item() for p in model.parameters())
    total = sum(p.numel()             for p in model.parameters())
    sparsity = zeros / total

    path = f"{SAVE_DIR}/pruned_{int(amount*100)}pct.pth"
    torch.save(model.state_dict(), path)

    prune_results.append({
        "Sparsity target": f"{int(amount*100)}%",
        "Actual sparsity": f"{sparsity*100:.1f}%",
        "Acc after prune (%)": round(acc_pruned*100, 2),
        "Acc after FT (%)":    round(acc_ft*100, 2),
        "Δ vs base (%)":       round((acc_ft - base_acc)*100, 2),
    })
    print(f"  Final: {acc_ft*100:.2f}%  actual sparsity={sparsity*100:.1f}%")
"""),
    ("code", """\
import pandas as pd
df = pd.DataFrame(prune_results).set_index("Sparsity target")
print("\\nPruning Summary")
print(df.to_string())
print(f"\\nBase accuracy (no pruning): {base_acc*100:.2f}%")
print("\\n⚠️  Reminder: sparsity does not reduce inference time on STM32.")
print("   See notebook 12 for deployment-relevant compression (INT8 quantization).")
"""),
]))


# ── 12 Quantization Pipeline ──────────────────────────
save("12_Quantization_Pipeline.ipynb", nb([
    ("markdown", """\
# 12 · Quantization Pipeline

Exports trained models to ONNX (FP32) and applies static INT8 QDQ quantization
for STM32 deployment via X-CUBE-AI.

**Why quantization matters for STM32 (unlike pruning):**
- INT8 reduces model size by ~4× (float32 → int8)
- STM32 Cortex-M cores with DSP extension execute INT8 MAC operations natively
- X-CUBE-AI generates optimized INT8 code from QDQ ONNX models
- Real speedup and memory reduction on target hardware

Run this notebook once per model variant you want to deploy.
Set MODEL_TAG to identify the variant in output filenames.
"""),
    *SETUP,
    ("code", """\
!pip -q install onnx onnxruntime onnxscript
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
import shutil, numpy as np
from pathlib import Path
from utils.dataset import manifest_paths, VWWDataset, get_eval_transform
from torch.utils.data import DataLoader
"""),
    SAVE_DIR_CELL,
    ("code", """\
# ── Configuration ─────────────────────────────────────────────────────
# Change MODEL_TAG and CKPT_PATH for each variant you export

MODEL_TAG  = "kd_ft"                                    # identifier for output files
CKPT_PATH  = f"{SAVE_DIR}/kd_ft_seed_XX.pth"           # checkpoint to export
MODEL_FN   = MobileNetV3_Pretrained                     # must match checkpoint

EXPORT_DIR    = Path(f"{SAVE_DIR}/exports_{MODEL_TAG}"); EXPORT_DIR.mkdir(exist_ok=True)
EVAL_SAMPLES  = 200
CALIB_SAMPLES = 200
IMG_SIZE      = 96

FP32_ONNX = f"vww_{MODEL_TAG}_fp32.onnx"
INT8_ONNX = f"vww_{MODEL_TAG}_int8_qdq.onnx"
CALIB_NPZ = f"vww_{MODEL_TAG}_calib_{CALIB_SAMPLES}.npz"
VAL_NPZ   = f"vww_{MODEL_TAG}_val_{EVAL_SAMPLES}.npz"
LABEL_NPZ = f"vww_{MODEL_TAG}_labels_{EVAL_SAMPLES}.npz"
"""),
    ("code", """\
# ── Load and verify model ─────────────────────────────────────────────
model = MODEL_FN().to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()

_, val_loader = get_loaders(batch_size=64)
acc = evaluate(model, val_loader, device)
print(f"PyTorch val accuracy: {acc*100:.2f}%  (checkpoint: {CKPT_PATH})")
"""),
    ("code", """\
# ── Export FP32 ONNX ──────────────────────────────────────────────────
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
torch.onnx.export(
    model, dummy, FP32_ONNX,
    input_names=["input"], output_names=["logits"],
    opset_version=18, do_constant_folding=True,
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    dynamo=False,
)
onnx.checker.check_model(FP32_ONNX)
print("✅ FP32 ONNX:", FP32_ONNX)
"""),
    ("code", """\
# ── Collect calibration and validation arrays ─────────────────────────
mp      = manifest_paths()
eval_tf = get_eval_transform()

val_ds   = VWWDataset(mp["val_person"],   mp["val_non_person"],   eval_tf)
train_ds = VWWDataset(mp["train_person"], mp["train_non_person"], eval_tf)

def collect(ds, n):
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    xs, ys = [], []
    for i, (x, y) in enumerate(loader):
        if i >= n: break
        xs.append(x.numpy().astype("float32")[0])
        ys.append(int(y.item()))
    return np.stack(xs), np.array(ys, dtype="int32")

val_x,   val_y   = collect(val_ds,   EVAL_SAMPLES)
calib_x, _       = collect(train_ds, CALIB_SAMPLES)

np.savez(VAL_NPZ,   input=val_x)
np.savez(LABEL_NPZ, label=val_y)
np.savez(CALIB_NPZ, input=calib_x)
print(f"Val:   {val_x.shape}  |  Calib: {calib_x.shape}")
"""),
    ("code", """\
# ── Static INT8 QDQ quantization ──────────────────────────────────────
class CalibReader(CalibrationDataReader):
    def __init__(self, npz, name="input"):
        self.x = np.load(npz)["input"]; self.name = name; self.i = 0
    def get_next(self):
        if self.i >= len(self.x): return None
        out = {self.name: self.x[self.i:self.i+1]}; self.i += 1; return out
    def rewind(self): self.i = 0

quantize_static(
    model_input=FP32_ONNX, model_output=INT8_ONNX,
    calibration_data_reader=CalibReader(CALIB_NPZ),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
    per_channel=True,
)
print("✅ INT8 QDQ ONNX:", INT8_ONNX)
"""),
    ("code", """\
# ── ONNX Runtime accuracy verification ───────────────────────────────
def onnx_accuracy(onnx_path, val_x, val_y):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    preds = [np.argmax(sess.run([out_name], {in_name: val_x[i:i+1]})[0][0])
             for i in range(len(val_x))]
    return (np.array(preds) == val_y).mean() * 100

fp32_ort_acc = onnx_accuracy(FP32_ONNX, val_x, val_y)
int8_ort_acc = onnx_accuracy(INT8_ONNX, val_x, val_y)
print(f"ONNX FP32 accuracy: {fp32_ort_acc:.2f}%")
print(f"ONNX INT8 accuracy: {int8_ort_acc:.2f}%")
print(f"INT8 accuracy drop: {fp32_ort_acc - int8_ort_acc:.2f}%")
"""),
    ("code", """\
# ── Copy all exports to Drive ─────────────────────────────────────────
for f in [FP32_ONNX, INT8_ONNX, CALIB_NPZ, VAL_NPZ, LABEL_NPZ]:
    if os.path.exists(f):
        shutil.copy2(f, EXPORT_DIR / f)
        print(f"✅ {f} → {EXPORT_DIR}")

print(f"\\nExport folder: {EXPORT_DIR}")
"""),
    ("markdown", """\
## STM32 deployment instructions

1. Open X-CUBE-AI in STM32CubeIDE
2. Import `vww_*_fp32.onnx` for FP32 inference
3. Import `vww_*_int8_qdq.onnx` for INT8 inference (recommended)
4. Use `vww_*_val_200.npz` as input batch for both validation runs
5. After each run, rename `network_val_io.npz`:
   - FP32 run → `stm32_outputs_fp32.npz`
   - INT8 run → `stm32_outputs_int8.npz`
6. Run `compute_stm32_accuracy()` below to score the results
"""),
    ("code", """\
def compute_stm32_accuracy(labels_npz, outputs_npz, key="c_outputs_1", num_classes=2):
    \"\"\"Score STM32 output NPZ against ground-truth labels.\"\"\"
    y      = np.load(labels_npz)["label"].astype("int64")
    raw    = np.load(outputs_npz)[key].reshape(len(y), num_classes)
    acc    = (np.argmax(raw, 1) == y).mean() * 100
    print(f"STM32 accuracy: {acc:.2f}%  ({len(y)} samples)")
    return acc

# Uncomment after STM32 runs:
# compute_stm32_accuracy(LABEL_NPZ, "stm32_outputs_fp32.npz")
# compute_stm32_accuracy(LABEL_NPZ, "stm32_outputs_int8.npz")
"""),
]))


# ══════════════════════════════════════════════════════
# PHASE 5: FINAL EVALUATION
# ══════════════════════════════════════════════════════

save("13_Final_Results.ipynb", nb([
    ("markdown", """\
# 13 · Final Results

**⚠️  This is the only notebook that uses the held-out test set.**
Do not load `get_test_loader()` anywhere else.

Aggregates all experiment results into unified comparison tables and plots.
Fill in checkpoint paths after completing notebooks 01–12.
"""),
    *SETUP,
    SAVE_DIR_CELL,
    ("code", """\
# ── Test loader — used ONLY here ─────────────────────────────────────
test_loader = get_test_loader(batch_size=64)
_, val_loader = get_loaders(batch_size=64)
"""),
    ("code", """\
# ── All checkpoints — fill in after running 01–12 ────────────────────
ALL_MODELS = [
    # (label, model_fn, checkpoint_path, phase)
    # Teachers
    ("VGG (scratch)",           VGG_Scratch,          f"{SAVE_DIR}/vgg_scratch_seed_XX.pth",          "Teacher"),
    ("VGG16-BN (pretrained)",   VGG_Pretrained,        f"{SAVE_DIR}/vgg_pretrained_seed_XX.pth",       "Teacher"),
    ("ResNet (scratch)",        ResNet_Scratch,         f"{SAVE_DIR}/resnet_scratch_seed_XX.pth",       "Teacher"),
    ("ResNet18 (pretrained)",   ResNet18_Pretrained,    f"{SAVE_DIR}/resnet18_pretrained_seed_XX.pth",  "Teacher"),
    # Students baseline
    ("MobileNetV2 (scratch)",          MobileNetV2_Scratch,    f"{SAVE_DIR}/mobilenetv2_baseline_seed_XX.pth", "Student"),
    ("MobileNetV3-Small (pretrained)", MobileNetV3_Pretrained, f"{SAVE_DIR}/mobilenetv3_baseline_seed_XX.pth", "Student"),
    # KD
    ("MobileNetV3 + KD-FT",     MobileNetV3_Pretrained, f"{SAVE_DIR}/kd_ft_seed_XX.pth",     "KD"),
    ("MobileNetV3 + KD-Scratch",MobileNetV3_Pretrained, f"{SAVE_DIR}/kd_scratch_seed_XX.pth","KD"),
    # Pruned
    ("MobileNetV3 + KD + Pruned 30%", MobileNetV3_Pretrained, f"{SAVE_DIR}/pruned_30pct.pth", "Pruned"),
]
"""),
    ("code", """\
def measure_latency(model, device, runs=200, warmup=20):
    model.eval()
    dummy = torch.randn(1, 3, 96, 96, device=device)
    with torch.no_grad():
        for _ in range(warmup): model(dummy)
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs): model(dummy)
        if device.type == "cuda": torch.cuda.synchronize()
    return (time.time() - t0) / runs * 1000

rows = []
for label, fn, ckpt, phase in ALL_MODELS:
    if not os.path.exists(ckpt):
        print(f"⚠️  Skipping: {label}"); continue
    m = fn().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    val_acc  = evaluate(m, val_loader,  device)
    test_acc = evaluate(m, test_loader, device)
    tp, _  = count_params(m)
    size   = model_size_mb(m)
    lat    = measure_latency(m, device)
    rows.append({
        "Model": label, "Phase": phase,
        "Val Acc (%)":  round(val_acc*100,  2),
        "Test Acc (%)": round(test_acc*100, 2),
        "Params (M)":   round(tp/1e6,       2),
        "Size (MB)":    round(size,          2),
        "Latency (ms)": round(lat,           3),
    })
    print(f"{label:42s}  val={val_acc*100:.2f}%  test={test_acc*100:.2f}%")
"""),
    ("code", """\
import pandas as pd
df = pd.DataFrame(rows).set_index("Model")
print("\\n" + "="*80)
print("FINAL RESULTS — TEST SET")
print("="*80)
print(df.drop(columns=["Phase"]).to_string())
"""),
    ("code", """\
# ── Accuracy comparison chart ─────────────────────────────────────────
phase_colors = {"Teacher": "#4878D0", "Student": "#55A868", "KD": "#E8735A", "Pruned": "#C44E52"}
colors = [phase_colors[r["Phase"]] for r in rows]

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(df.index, df["Test Acc (%)"], color=colors, edgecolor="white")
ax.set_ylabel("Test Accuracy (%)"); ax.set_title("Final Results — Test Set Accuracy")
ax.set_ylim(df["Test Acc (%)"].min() - 3, 100)
for p in bars:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x()+p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=7)

from matplotlib.patches import Patch
legend = [Patch(color=c, label=l) for l, c in phase_colors.items()]
ax.legend(handles=legend, loc="lower right")
plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.show()
"""),
    ("code", """\
# ── Accuracy vs Model Size scatter ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for _, row in df.iterrows():
    ax.scatter(row["Size (MB)"], row["Test Acc (%)"],
               color=phase_colors.get(row["Phase"], "gray"), s=80, zorder=3)
    ax.annotate(row.name, (row["Size (MB)"], row["Test Acc (%)"]),
                textcoords="offset points", xytext=(5, 3), fontsize=7)
ax.set_xlabel("Model Size (MB)"); ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Accuracy vs Model Size")
ax.legend(handles=legend); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
"""),
    ("code", """\
# ── KD improvement summary ────────────────────────────────────────────
print("\\nKnowledge Distillation Impact")
print("─"*50)
if "MobileNetV3-Small (pretrained)" in df.index and "MobileNetV3 + KD-FT" in df.index:
    baseline = df.loc["MobileNetV3-Small (pretrained)", "Test Acc (%)"]
    kd_ft    = df.loc["MobileNetV3 + KD-FT",           "Test Acc (%)"]
    print(f"  Student baseline (no KD):  {baseline:.2f}%")
    print(f"  Student + KD-FT:           {kd_ft:.2f}%")
    print(f"  Improvement:               +{kd_ft - baseline:.2f}%")

print("\\nQuantization Impact (see notebook 12 for ONNX accuracy)")
print("─"*50)
print("  FP32 ONNX:  see notebook 12")
print("  INT8 ONNX:  see notebook 12")
print("  STM32 FP32: fill in after hardware validation")
print("  STM32 INT8: fill in after hardware validation")
"""),
]))


print(f"\nAll 13 notebooks generated in: {OUT}")
