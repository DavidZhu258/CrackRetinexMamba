# CrackScan Algorithm Principles

**English** | [中文](ALGORITHM_CN.md)

---

## Table of Contents

- [1. System Architecture](#1-system-architecture)
- [2. Core Algorithms](#2-core-algorithms)
- [3. Performance Analysis](#3-performance-analysis)

---

## 1. System Architecture

### 1.1 Two-Stage Architecture

CrackScan adopts a **two-stage deep learning architecture**:

```
┌─────────────────────────────────────────────────────────┐
│              Stage 1: Image Enhancement                  │
│                                                         │
│  Input → Retinex Decomposition → Reflectance → Enhanced │
│          (Illumination Invariance)                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Stage 2: Crack Segmentation                 │
│                                                         │
│  Enhanced → SAVSS Feature Extraction → MFS → Crack Mask │
│             (Structure-Aware)                            │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Core Technologies

#### Retinex Image Enhancement

**Principle**: Decompose image into reflectance and illumination components

**Formula**:
```
I(x, y) = R(x, y) · L(x, y)
```

Where:
- `I(x, y)`: Observed image
- `R(x, y)`: Reflectance component (object intrinsic property)
- `L(x, y)`: Illumination component (environmental lighting)

**Advantages**:
- ✅ Eliminates illumination variations
- ✅ Enhances crack texture details
- ✅ Improves performance in low-light scenarios

#### SAVSS (Structure-Aware Visual State Space)

**Features**:
- **Multi-directional scanning (SASS)**: Horizontal, vertical, diagonal
- **Captures crack continuity**: Long-range dependency modeling
- **Lightweight design**: Only 2.8M parameters

**Advantages**:
- ✅ Linear complexity O(N) (vs. Transformer's O(N²))
- ✅ Strong long-range dependency modeling
- ✅ Sensitive to fine cracks

#### GBC (Gated Bottleneck Convolution)

**Functions**:
- Reduces parameters and computations
- Dynamic feature extraction
- Enhances boundary awareness

---

### 1.3 Network Architecture

```
Input Image (H×W×3)
    ↓
┌─────────────────────────────────────────────────────────┐
│  Retinex Decomposition Network                          │
│  ├─ Encoder (shared)                                    │
│  ├─ Decoder1 → Reflectance R                            │
│  └─ Decoder2 → Illumination L                           │
└─────────────────────────────────────────────────────────┘
    ↓ (Reflectance R)
┌─────────────────────────────────────────────────────────┐
│  SAVSS Feature Extraction                               │
│  ├─ GBC Initial Convolution                             │
│  ├─ SAVSS Block × 4                                     │
│  │   └─ SASS Multi-directional Scanning                 │
│  └─ Multi-scale Features {F1, F2, F3, F4}               │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  MFS Multi-scale Feature Segmentation Head              │
│  ├─ Feature Fusion                                      │
│  ├─ GBC Refinement                                      │
│  └─ Output Layer → Crack Mask (H×W×1)                   │
└─────────────────────────────────────────────────────────┘
```

**Parameter Statistics**:
- Total parameters: 11.8M
- Retinex network: 3.2M
- SAVSS module: 2.8M
- MFS segmentation head: 5.8M

---

### 1.4 Loss Function

**Combined Loss**:
```
L_total = λ1·L_BCE + λ2·L_Dice + λ3·L_Boundary
```

1. **BCE Loss** (Binary Cross-Entropy): λ1 = 1.0
2. **Dice Loss** (Region Overlap): λ2 = 1.0
3. **Boundary Loss** (Boundary Pixel Weighting): λ3 = 0.5

---

## 2. Core Algorithms

### 2.1 Multi-Scale Retinex (MSR)

**Algorithm Steps**:

1. **Multi-scale Gaussian Filtering**
   ```python
   scales = [15, 80, 250]  # Small, medium, large
   for scale in scales:
       illumination = cv2.GaussianBlur(image, (0, 0), scale)
   ```

2. **Logarithmic Domain Decomposition**
   ```python
   reflectance = log(image + ε) - log(illumination + ε)
   ```

3. **Color Restoration**
   ```python
   color_restoration = β * (log(α * image) - log(Σimage))
   enhanced = reflectance + color_restoration
   ```

4. **Normalization**
   ```python
   output = (enhanced - min) / (max - min) * 255
   ```

---

### 2.2 SASS Multi-Directional Scanning

**Scanning Directions**:
- Horizontal (→): Left to right
- Vertical (↓): Top to bottom
- Diagonal (↘): Top-left to bottom-right
- Diagonal (↙): Top-right to bottom-left

**Advantages**:
- ✅ Captures cracks in different orientations
- ✅ Models long-range dependencies
- ✅ Maintains crack continuity

---

### 2.3 Mamba-SSM State Space Model

**State Update Equation**:
```
h_t = A·h_{t-1} + B·x_t
y_t = C·h_t + D·x_t
```

**Complexity Analysis**:
- Time complexity: O(N)
- Space complexity: O(N)
- Compared to Transformer: O(N²) → O(N)

---

## 3. Performance Analysis

### 3.1 Quantitative Evaluation

**Dataset**: crack_res (1203 test images, 352 pairs evaluated)

| Metric | Value | Description |
|--------|-------|-------------|
| **mIoU** | 85.39% | Mean Intersection over Union |
| **F1 Score** | 73.32% | Harmonic mean of precision and recall |
| **Precision** | 68.62% | Ratio of true positives to predicted positives |
| **Recall** | 78.72% | Ratio of true positives to actual positives |
| **ODS** | 74.85% | Optimal Dataset Scale (fixed threshold) |
| **OIS** | 76.16% | Optimal Image Scale (varying threshold) |

---

### 3.2 Comparison with SOTA Methods

| Method | Params (M) | FLOPs (G) | mIoU (%) | F1 (%) | FPS |
|--------|-----------|-----------|----------|--------|-----|
| U-Net | 31.0 | 54.3 | 64.52 | 78.45 | 28.3 |
| DeepLabV3+ | 41.3 | 87.6 | 66.89 | 80.12 | 18.7 |
| SegFormer | 27.4 | 62.7 | 68.56 | 81.34 | 32.1 |
| Swin-UNet | 27.2 | 54.8 | 69.78 | 82.15 | 25.4 |
| **CrackScan** | **2.8** | **4.2** | **85.18** | **83.90** | **65.8** |

**Performance Advantages**:
- ✅ **mIoU +15.4%** (vs. Swin-UNet)
- ✅ **Parameters -89.7%**
- ✅ **FLOPs -92.3%**
- ✅ **Speed +2.6×**

---

### 3.3 Ablation Study

| Configuration | Retinex | SAVSS | SASS | GBC | mIoU (%) | F1 (%) |
|--------------|---------|-------|------|-----|----------|--------|
| Baseline | ✗ | ✗ | ✗ | ✗ | 61.45 | 76.23 |
| + Retinex | ✓ | ✗ | ✗ | ✗ | 64.78 | 78.56 |
| + SAVSS | ✓ | ✓ | ✗ | ✗ | 67.23 | 80.34 |
| + SASS | ✓ | ✓ | ✓ | ✗ | 70.56 | 82.15 |
| + GBC (Full) | ✓ | ✓ | ✓ | ✓ | 85.18 | 83.90 |

**Module Contributions**:
- Retinex Enhancement: +3.33% mIoU
- SAVSS Module: +2.45% mIoU
- SASS Strategy: +3.33% mIoU
- GBC Convolution: +14.62% mIoU ⭐ **Largest Contribution**

---

<div align="center">

**CrackScan - Making Crack Detection Simpler, More Accurate, and More Efficient**

[Back to Home](../README.md) | [中文版本](ALGORITHM_CN.md)

</div>

