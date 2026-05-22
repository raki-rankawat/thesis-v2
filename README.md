# Optimization of Deep Learning Models for Edge Deployment
### MSc Thesis — IEBI Lab, Università degli Studi di Milano

> Supervisor: angelo.genovese@unimib.it

This repository contains the full experimental pipeline for my MSc thesis on compressing CNN models for deployment on the STM32 microcontroller using knowledge distillation, pruning, and INT8 quantization — evaluated on the Visual Wake Words (VWW) binary classification task.

---

## Research Question

> *Can knowledge distillation from a strong pretrained teacher, followed by quantization, compress a CNN to run on STM32 while preserving acceptable accuracy on the Visual Wake Words task?*

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1 — Teacher Selection                                    │
│  Train 4 candidates → compare → select best teacher            │
│  VGG (scratch) · VGG16-BN (pretrained)                         │
│  ResNet (scratch) · ResNet18 (pretrained) ← selected           │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2 — Student Selection (2×2 controlled design)           │
│  Train all 4 combinations → separate architecture vs init      │
│                  │  Scratch      │  Pretrained                 │
│  MobileNetV2     │  nb 06        │  nb 07                      │
│  MobileNetV3     │  nb 08        │  nb 09 ← selected           │
├─────────────────────────────────────────────────────────────────┤
│  Phase 3 — Knowledge Distillation                               │
│  Hyperparameter ablation (T, α) → KD-FT vs KD-Scratch          │
├─────────────────────────────────────────────────────────────────┤
│  Phase 4 — Compression                                          │
│  Pruning (additional) · INT8 Quantization (deployment target)  │
├─────────────────────────────────────────────────────────────────┤
│  Phase 5 — Final Evaluation                                     │
│  Held-out test set · ONNX FP32/INT8 · STM32 hardware           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dataset

**Visual Wake Words (VWW)** — binary classification: `person` / `non_person`  
Source: [SiLabs VWW Dataset](https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz) (derived from COCO 2014)

| Split | Images | Per class |
|-------|--------|-----------|
| Train | 7,000  | 3,500     |
| Val   | 1,500  | 750       |
| Test  | 1,500  | 750       |

- Input resolution: 96×96 RGB
- Splits are **fixed** via seeded manifests (seed=41) — fully reproducible
- Test split is **held out** and only evaluated in notebook 15

---

## Models

### Teachers

| Model | Init | Params | Role |
|-------|------|--------|------|
| VGG (scratch) | Random | ~6.0M | Comparison baseline |
| VGG16-BN | ImageNet | ~138M | Comparison candidate |
| ResNet (scratch) | Random | ~2.8M | Comparison baseline |
| **ResNet18** | **ImageNet** | **~11.2M** | **Selected teacher** |

ResNet18 selected: best accuracy/size tradeoff, residual connections generalize better than VGG on small datasets, progressive unfreeze avoids overfitting.

### Students — 2×2 Controlled Design

| | Scratch | Pretrained |
|---|---|---|
| **MobileNetV2** | Custom implementation | torchvision MobileNetV2 |
| **MobileNetV3-Small** | torchvision (weights=None) | **torchvision (selected)** |

The 2×2 design isolates two independent effects:

- **Architecture effect** — V3 vs V2, controlling for initialization  
- **Initialization effect** — pretrained vs scratch, controlling for architecture

MobileNetV3-Small (pretrained) is the selected student: wins on both dimensions, hardware-aware NAS design, SE blocks, h-swish activations, suitable for STM32 deployment after quantization.

---

## Knowledge Distillation

Uses Hinton et al. (2015) soft-target distillation loss:

$$\mathcal{L} = \alpha \cdot T^2 \cdot \text{KL}(\sigma(z_s/T) \| \sigma(z_t/T)) + (1-\alpha) \cdot \text{CE}(z_s, y)$$

**Hyperparameter ablation** (notebook 11):

| Config | T | α | Expected effect |
|--------|---|---|-----------------|
| Low temperature | 2 | 0.7 | Less soft information |
| Balanced | 4 | 0.5 | Equal hard/soft weight |
| **Default** | **4** | **0.7** | **Selected config** |
| Soft-heavy | 4 | 0.9 | Student relies more on teacher |
| High temperature | 8 | 0.7 | Maximum knowledge transfer |

**Two initialization modes** compared in notebook 12:
- **KD-FT**: student starts from pretrained baseline, then KD
- **KD-Scratch**: student starts from random init, then KD

---

## Compression & Deployment

### Pruning (notebook 13)
Unstructured L1 magnitude pruning at 20%, 30%, 50% sparsity with short fine-tuning.  
**Note:** unstructured sparsity does not reduce inference latency on STM32 — X-CUBE-AI executes dense operations. Included to measure accuracy degradation vs sparsity; quantization is the deployment-relevant technique.

### Quantization (notebook 14)
Static INT8 QDQ quantization via ONNX Runtime:
- FP32 ONNX export → calibration on 200 training samples → INT8 QDQ ONNX
- ~4× size reduction, native INT8 MAC acceleration on STM32 Cortex-M with DSP
- Validated on ONNX Runtime before STM32 deployment

### STM32 Deployment
Import INT8 QDQ ONNX into X-CUBE-AI (STM32CubeIDE), run validation with the provided `.npz` input arrays, score outputs with `compute_stm32_accuracy()` in notebook 14.

---

## Standardized Training Conditions

All models trained under identical conditions — only architecture and initialization vary:

| Parameter | Scratch | Pretrained |
|-----------|---------|------------|
| Batch size | 64 | 64 |
| Optimizer | Adam | Adam |
| Weight decay | 1e-4 | 1e-4 |
| Label smoothing | 0.1 | 0.1 |
| Augmentation | standard | standard |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Early stopping patience | 10 | 10 |
| Seeds | [41, 52, 63] | [41, 52, 63] |
| Max epochs | 50 | 25–30 |
| LR | 1e-3 | 3e-4 → 1e-4 (progressive) |

Results reported as **mean ± std across 3 seeds** for statistical credibility.

---

## Notebook Execution Order

Run strictly in order — each notebook depends on checkpoints from the previous phase.

```
Phase 1 — Teacher Selection
  01_Teacher_VGG_Scratch.ipynb
  02_Teacher_VGG_Pretrained.ipynb
  03_Teacher_ResNet_Scratch.ipynb
  04_Teacher_ResNet18_Pretrained.ipynb
  05_Teacher_Comparison.ipynb            ← pick best teacher

Phase 2 — Student Selection
  06_Student_MobileNetV2_Scratch.ipynb
  07_Student_MobileNetV2_Pretrained.ipynb
  08_Student_MobileNetV3_Scratch.ipynb
  09_Student_MobileNetV3_Pretrained.ipynb
  10_Student_Comparison.ipynb            ← 2×2 analysis, pick best student

Phase 3 — Knowledge Distillation
  11_KD_Ablation.ipynb                   ← hyperparameter sensitivity
  12_KD_Main.ipynb                       ← KD-FT vs KD-Scratch, 3 seeds

Phase 4 — Compression
  13_Pruning.ipynb                       ← sparsity sweep (additional experiment)
  14_Quantization_Pipeline.ipynb         ← FP32 + INT8 ONNX export

Phase 5 — Final Evaluation
  15_Final_Results.ipynb                 ← test set only, all tables, all plots
```

> ⚠️ `get_test_loader()` is called **only** in notebook 15. Never use it during training or model selection.

---

## Repository Structure

```
thesis/
├── utils/
│   ├── dataset.py          # VWW download, manifests, train/val/test loaders
│   ├── models.py           # All 8 model definitions + registries
│   └── train.py            # Training loops, KD loss, multi-seed runner
│
├── 01–05  Teacher notebooks
├── 06–10  Student notebooks
├── 11–12  KD notebooks
├── 13–14  Compression notebooks
├── 15     Final results
│
├── generate_notebooks.py         # Regenerates all notebooks from source
├── generate_student_notebooks.py # Regenerates student notebooks only
└── README.md
```

Checkpoints (`.pth`), ONNX exports (`.onnx`), and data arrays (`.npz`) are **not committed** — they live on Google Drive. See `.gitignore`.

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Upload utils to Google Drive
Place the `utils/` folder at:
```
My Drive/thesis/utils/
```
Every notebook mounts Drive and copies utils to `/content/utils` at startup.

### 3. Open any notebook in Google Colab
Each notebook is self-contained after the utils copy step. No local dependencies required.

### 4. Run in order
Start from `01_Teacher_VGG_Scratch.ipynb`. After each notebook, update the `seed_XX` placeholders in the next notebook's config cell with your best seed number.

---

## Dependencies

All dependencies are standard Colab packages. The quantization pipeline additionally requires:

```bash
pip install onnx onnxruntime onnxscript
```

This is handled automatically in notebook 14.

| Package | Version |
|---------|---------|
| PyTorch | ≥ 2.0 |
| torchvision | ≥ 0.15 |
| onnx | ≥ 1.14 |
| onnxruntime | ≥ 1.16 |
| numpy | ≥ 1.24 |
| matplotlib | ≥ 3.7 |
| pandas | ≥ 2.0 |

---

## .gitignore

```gitignore
# Checkpoints — live on Google Drive
*.pth

# ONNX exports
*.onnx
*.npz

# Jupyter artifacts
.ipynb_checkpoints/
*/.ipynb_checkpoints/

# Python cache
__pycache__/
*.pyc
*.pyo

# Dataset — download via prepare_dataset()
vww_work/
*.tar.gz
```

---

## Citation

If you use this code or pipeline, please cite:

```bibtex
@mastersthesis{yourname2025,
  title   = {Optimization of Deep Learning Models for Edge Deployment},
  author  = {Your Name},
  school  = {Università degli Studi di Milano},
  year    = {2025},
  note    = {IEBI Lab, supervisor: Angelo Genovese}
}
```

Hinton et al. (2015) knowledge distillation:
```bibtex
@article{hinton2015distilling,
  title   = {Distilling the Knowledge in a Neural Network},
  author  = {Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal = {arXiv preprint arXiv:1503.02531},
  year    = {2015}
}
```
