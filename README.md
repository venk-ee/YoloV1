# YOLOv1 — From-Scratch PyTorch Implementation

A complete, from-scratch implementation of **YOLOv1** (*You Only Look Once: Unified, Real-Time Object Detection*) in PyTorch — engineered to train on a **single consumer-grade GPU**.

---

## Why This Project?

The original YOLOv1 paper pre-trains its backbone on ImageNet-1K before fine-tuning on detection tasks, requiring significant compute resources. This implementation takes a different approach:

| Design Decision | Original Paper | This Implementation |
|---|---|---|
| Backbone pre-training | ImageNet-1K (1000 classes) | **None — trained end-to-end from scratch** |
| FC layer width | 4096 neurons | **496 neurons** (≈ 8× reduction) |
| Target hardware | Multi-GPU cluster | **Single low-end GPU** |
| Dataset | VOC + ImageNet | **PASCAL VOC only** |

> The fully connected layer was intentionally reduced from 4096 → 496 neurons to dramatically lower VRAM usage and parameter count, enabling training on hardware with a single GPU.

---

## Architecture Overview

The model follows the Darknet-inspired convolutional backbone described in the YOLOv1 paper, with modifications to the detection head:

```
Input Image (448×448×3)
        │
        ▼
┌─────────────────────┐
│   24 Conv Layers     │   Backbone: Conv → BatchNorm → LeakyReLU(0.1)
│   (Darknet-style)    │   with MaxPool downsampling
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Fully Connected    │   Flatten → Linear(1024·S·S, 496) → Dropout(0.5)
│   Detection Head     │   → LeakyReLU(0.1) → Linear(496, S·S·30)
└─────────────────────┘
        │
        ▼
  Output: 7×7×30 tensor
  (class probabilities + bounding box predictions)
```

**Grid configuration:** `S=7`, `B=2` boxes per cell, `C=20` PASCAL VOC classes.

---

## Project Structure

```
├── model.py        # YOLOv1 architecture (Darknet backbone + detection head)
├── train.py        # Training loop with mAP evaluation per epoch
├── loss.py         # Multi-part YOLO loss (coord + object + no-object + class)
├── dataset.py      # Custom PyTorch Dataset for PASCAL VOC format
├── utils.py        # IoU, NMS, mAP, bbox conversion, and visualization
└── README.md
```

### Module Breakdown

- **`model.py`** — Defines `YOLOV1` and `CNNBlock` modules. The architecture is driven by a declarative config list (`architecture_config`), making the layer structure easy to read and modify.
- **`train.py`** — Orchestrates end-to-end training: data loading, forward/backward pass with Adam optimizer, and per-epoch mAP tracking on the training set.
- **`loss.py`** — Implements the multi-part YOLOv1 loss function with λ-weighted terms for coordinate regression, objectness confidence, no-object penalty, and class prediction (MSE-based).
- **`dataset.py`** — Loads PASCAL VOC-format annotations and encodes ground-truth bounding boxes into the `S×S×30` label matrix required by the network.
- **`utils.py`** — A collection of core object detection utilities:
  - `intersection_over_union()` — IoU computation (midpoint and corner formats)
  - `non_max_suppression()` — Post-processing to filter redundant detections
  - `mean_average_precision()` — mAP evaluation across all 20 classes
  - `convert_cellboxes()` / `cellboxes_to_boxes()` — Grid-relative → image-relative coordinate conversion
  - `plot_image()` — Bounding box visualization with Matplotlib
  - `save_checkpoint()` / `load_checkpoint()` — Model persistence

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch ≥ 1.9 with CUDA support
- torchvision, pandas, numpy, matplotlib, tqdm

```bash
pip install torch torchvision pandas numpy matplotlib tqdm
```

### Dataset Setup

1. Download the **PASCAL VOC** dataset (images + label `.txt` files in YOLO format).
2. Prepare CSV annotation files mapping image filenames to label filenames.
3. Update `IMG_DIR` and `LABEL_DIR` in `train.py` to point to your dataset paths.

### Train

```bash
python train.py
```

### Hyperparameters

All training hyperparameters are defined at the top of `train.py` for easy tuning:

| Parameter | Default | Description |
|---|---|---|
| `LEARNING_RATE` | `2e-5` | Adam optimizer learning rate |
| `BATCH_SIZE` | `8` | Samples per batch (lower if GPU OOM) |
| `EPOCH` | `100` | Total training epochs |
| `NUM_WORKERS` | `2` | DataLoader parallel workers |
| `WEIGHT_DECAY` | `0` | L2 regularization factor |

---

## Loss Function

The loss in `loss.py` follows the original YOLOv1 formulation with four components:

```
Loss = λ_coord · (box coordinate loss)
     + (object confidence loss)
     + λ_noobj · (no-object confidence loss)
     + (class prediction loss)
```

Where `λ_coord = 5` and `λ_noobj = 0.5`, as specified in the paper. Square root is applied to width/height predictions for scale normalization.

---

## Acknowledgments

- **YOLOv1 Paper:** *You Only Look Once: Unified, Real-Time Object Detection* — J. Redmon et al., CVPR 2016.
- Utility functions in `utils.py` were adapted from various open-source PyTorch object detection resources.

---

*Built with PyTorch 🔥*
