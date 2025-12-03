#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同权重配置的加权mIoU效果
"""

import numpy as np
from PIL import Image
import os
import sys

def cal_weighted_mIoU(pred_list, gt_list, w_fg=0.7, w_bg=0.3, thresh_step=0.01):
    """
    加权mIoU计算
    
    Args:
        pred_list: 预测图像列表
        gt_list: 真值图像列表
        w_fg: 前景权重
        w_bg: 背景权重
        thresh_step: 阈值步长
    
    Returns:
        mIoU: 最大加权mIoU值
    """
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            
            TP = np.sum((pred_img == 1) & (gt_img == 1))
            TN = np.sum((pred_img == 0) & (gt_img == 0))
            FP = np.sum((pred_img == 1) & (gt_img == 0))
            FN = np.sum((pred_img == 0) & (gt_img == 1))
            
            if (FN + FP + TP) <= 0:
                iou = 0
            else:
                iou_fg = TP / (FN + FP + TP)  # 前景IoU
                iou_bg = TN / (FN + FP + TN)  # 背景IoU
                iou = w_fg * iou_fg + w_bg * iou_bg  # 加权
            
            iou_list.append(iou)
        
        ave_iou = np.mean(np.array(iou_list))
        final_iou.append(ave_iou)
    
    mIoU = np.max(np.array(final_iou))
    return mIoU


def get_image_pairs(results_dir, suffix_gt="lab", suffix_pred="pre"):
    """获取图像对"""
    pred_imgs = []
    gt_imgs = []
    
    pred_dir = os.path.join(results_dir, suffix_pred)
    gt_dir = os.path.join(results_dir, suffix_gt)
    
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg'))])
    
    for pred_file in pred_files:
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, pred_file)
        
        if os.path.exists(gt_path):
            pred_img = np.array(Image.open(pred_path).convert('L'))
            gt_img = np.array(Image.open(gt_path).convert('L'))
            pred_imgs.append(pred_img)
            gt_imgs.append(gt_img)
    
    return pred_imgs, gt_imgs


if __name__ == '__main__':
    results_dir = "../results/results_test/2025_10_11_16:19:24_Dataset->MID_original"
    
    print("=" * 80)
    print("加权mIoU测试 - 不同权重配置对比")
    print("=" * 80)
    print()
    
    print("加载图像...")
    pred_list, gt_list = get_image_pairs(results_dir)
    print(f"✅ 找到 {len(pred_list)} 对图像")
    print()
    
    # 测试不同权重配置
    weight_configs = [
        (0.5, 0.5, "原始方案 (平均)"),
        (0.6, 0.4, "轻微偏向前景"),
        (0.7, 0.3, "推荐配置 ✅"),
        (0.8, 0.2, "强调前景"),
        (0.9, 0.1, "极度强调前景"),
        (1.0, 0.0, "仅前景IoU"),
    ]
    
    print("=" * 80)
    print("测试不同权重配置...")
    print("=" * 80)
    print()
    
    results = []
    for w_fg, w_bg, desc in weight_configs:
        print(f"测试: {desc} (前景{w_fg:.1f} : 背景{w_bg:.1f})")
        miou = cal_weighted_mIoU(pred_list, gt_list, w_fg, w_bg)
        results.append((w_fg, w_bg, miou, desc))
        print(f"  mIoU = {miou:.4f} ({miou*100:.2f}%)")
        print()
    
    # 找出最佳配置
    best_config = max(results, key=lambda x: x[2])
    
    print("=" * 80)
    print("结果汇总")
    print("=" * 80)
    print()
    print(f"{'权重配置':<20} {'mIoU':<12} {'百分比':<10} {'说明'}")
    print("-" * 80)
    for w_fg, w_bg, miou, desc in results:
        marker = " ⭐" if (w_fg, w_bg) == (best_config[0], best_config[1]) else ""
        print(f"{w_fg:.1f}:{w_bg:.1f} (前景:背景)  {miou:.4f}      {miou*100:.2f}%     {desc}{marker}")
    print()
    
    print("=" * 80)
    print("分析与建议")
    print("=" * 80)
    print()
    print(f"🏆 最佳配置: 前景{best_config[0]:.1f} : 背景{best_config[1]:.1f}")
    print(f"   mIoU = {best_config[2]:.4f} ({best_config[2]*100:.2f}%)")
    print(f"   说明: {best_config[3]}")
    print()
    
    # 对比原始方案
    original_miou = results[0][2]  # 0.5:0.5
    improvement = (best_config[2] - original_miou) * 100
    
    print(f"📊 相比原始方案 (0.5:0.5):")
    print(f"   原始mIoU: {original_miou:.4f} ({original_miou*100:.2f}%)")
    print(f"   最佳mIoU: {best_config[2]:.4f} ({best_config[2]*100:.2f}%)")
    print(f"   提升幅度: {improvement:+.2f}% (绝对值)")
    print()
    
    print("💡 建议:")
    if best_config[0] >= 0.8:
        print("   - 您的数据集更适合强调前景(裂缝)检测")
        print("   - 建议使用权重 0.8:0.2 或更高")
    elif best_config[0] >= 0.6:
        print("   - 您的数据集适合平衡前景和背景")
        print("   - 建议使用权重 0.7:0.3 (推荐配置)")
    else:
        print("   - 您的数据集适合平衡评估")
        print("   - 建议使用权重 0.5:0.5 (原始方案)")
    print()
    
    print("=" * 80)
    print("如何应用最佳配置:")
    print("=" * 80)
    print()
    print("在 evaluate_1.py 第162行函数中修改:")
    print(f"    w_foreground = {best_config[0]:.1f}  # 前景权重")
    print(f"    w_background = {best_config[1]:.1f}  # 背景权重")
    print()
    print("=" * 80)

