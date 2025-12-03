# CrackScan - 裂隙智能检测系统

<div align="center">

**基于深度学习的高精度裂隙检测系统**

*轻量级模型 · 高准确率 · 图像增强 · 多数据集验证*

**中文** | [English](README.md)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.6-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Academic-red.svg)](LICENSE)

</div>

---

## 🌟 核心亮点

- ✅ **轻量级模型**: 仅 45.2 MB，11.8M 参数
- ✅ **高准确率**: mIoU 85.4%，超越传统方法
- ✅ **图像增强**: Retinex 算法，低光照场景提升 15-20%
- ✅ **多数据集验证**: 在混凝土、金属、木材等多种材料上验证
- ✅ **快速推理**: 单张图片 < 2 秒（GPU）
- ✅ **批量处理**: 自动处理多文件夹

---

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| **mIoU** | 85.4% |
| **Precision** | 92.3% |
| **Recall** | 89.7% |
| **F1-Score** | 90.9% |
| **模型大小** | 45.2 MB |
| **推理速度** | 1.8 秒/张 |

---

## 📸 检测结果展示

### 混凝土裂缝

<table>
<tr>
<td align="center"><b>原始图片</b></td>
<td align="center"><b>预测掩码</b></td>
<td align="center"><b>可视化结果</b></td>
</tr>
<tr>
<td><img src="data/prediction_results/kaggle_cracks/00027_original.jpg" width="250"/></td>
<td><img src="data/prediction_results/kaggle_cracks/00027_prediction.png" width="250"/></td>
<td><img src="data/prediction_results/kaggle_cracks/00027_visualization.jpg" width="250"/></td>
</tr>
</table>

### 金属表面缺陷

<table>
<tr>
<td align="center"><b>原始图片</b></td>
<td align="center"><b>预测掩码</b></td>
<td align="center"><b>可视化结果</b></td>
</tr>
<tr>
<td><img src="data/prediction_results/kaggle_iron/3d6fc6cb2_original.jpg" width="250"/></td>
<td><img src="data/prediction_results/kaggle_iron/3d6fc6cb2_prediction.png" width="250"/></td>
<td><img src="data/prediction_results/kaggle_iron/3d6fc6cb2_visualization.jpg" width="250"/></td>
</tr>
</table>

### 木材表面缺陷

<table>
<tr>
<td align="center"><b>原始图片</b></td>
<td align="center"><b>预测掩码</b></td>
<td align="center"><b>可视化结果</b></td>
</tr>
<tr>
<td><img src="data/prediction_results/kaggle_wood/99900054_original.jpg" width="250"/></td>
<td><img src="data/prediction_results/kaggle_wood/99900054_prediction.png" width="250"/></td>
<td><img src="data/prediction_results/kaggle_wood/99900054_visualization.jpg" width="250"/></td>
</tr>
</table>

---

## 🚀 快速开始

### 环境安装

```bash
# 创建环境
conda create -n CrackScan python=3.10 -y
conda activate CrackScan

# 安装 PyTorch
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# 安装 MMCV
python -m pip install -U openmim
mim install mmcv-full

# 安装依赖
python -m pip install mamba-ssm==1.2.0
python -m pip install timm lmdb mmengine
pip install transformers==4.31.0
pip install numpy==1.23.5
pip install -U scikit-learn scikit-image
```

### 运行检测

```bash
cd commit

# 启用图像增强（推荐）
python predict_simple.py --enhance \
  --input_dir data/cut_picture \
  --output_dir data/prediction_results
```

---

## 📚 文档

- [📖 中文完整文档](docs/USER_MANUAL_CN.md) - 完整使用手册
- [📖 English Documentation](docs/USER_MANUAL_EN.md) - Complete user manual
- [🔬 算法原理](docs/ALGORITHM_CN.md) - 技术细节
- [🔬 Algorithm Principles](docs/ALGORITHM.md) - Technical details

---

## 🎯 应用场景

- 🏗️ **建筑工程**: 混凝土结构裂缝检测
- 🏭 **工业检测**: 金属表面缺陷识别
- 🌉 **基础设施**: 桥梁、道路裂缝监测
- 🪵 **材料科学**: 木材、复合材料缺陷分析

---

## 📈 数据集验证

可以在此下载[Google Drive](https://drive.google.com/drive/folders/17i3EkYEs00Jwmxm5MLc_sKXNYPlbyiwO?hl=zh-cn)

| 数据集 | 类型 | 图片数量 | mIoU | 说明 |
|--------|------|----------|------|------|
| **Kaggle Cracks** | 混凝土裂缝 | 13 | 88.5% | 建筑裂缝检测 |
| **Kaggle Iron** | 金属缺陷 | 500+ | 82.3% | 工业缺陷检测 |
| **Kaggle Wood** | 木材缺陷 | 100+ | 84.1% | 木材表面分析 |
| **Rock Cracks** | 岩石裂隙 | 352 对 | 85.4% | 地质裂隙分析 |

---

## 📞 联系方式

**作者**: David Zhu  
**版本**: v1.0  
**日期**: 2025-12-02

---

<div align="center">

**CrackScan - 让裂隙检测更简单、更准确、更高效**

⭐ 如果这个项目对您有帮助，请给我们一个 Star！⭐

</div>

