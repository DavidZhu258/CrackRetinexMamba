#!/usr/bin/env python3
"""
简化版裂隙预测脚本
- 不检测尺子
- 不计算三大指标
- 只预测裂隙区域
- 支持批量处理子文件夹
"""

import numpy as np
import torch
import argparse
import os
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

from models import build_model
from main import get_args_parser


def retinex_enhancement(image):
    """
    Retinex 图像增强

    Args:
        image: 输入图像 (numpy array, BGR)

    Returns:
        enhanced: 增强后的图像 (numpy array, BGR)
    """
    try:
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        image_float = image.astype(np.float32) / 255.0

        # 多尺度Retinex (MSR)
        scales = [15, 80, 250]  # 小、中、大尺度
        msr_result = np.zeros_like(image_float)

        for scale in scales:
            # 高斯模糊估计光照分量
            illumination = cv2.GaussianBlur(image_float, (0, 0), scale)
            illumination = np.maximum(illumination, 0.01)  # 避免除零

            # 计算反射分量: R = log(I) - log(L)
            reflectance = np.log(image_float + 0.01) - np.log(illumination + 0.01)
            msr_result += reflectance

        # 多尺度平均
        msr_result = msr_result / len(scales)

        # 颜色恢复 (Color Restoration)
        sum_channels = np.sum(image_float, axis=2, keepdims=True)
        sum_channels = np.maximum(sum_channels, 0.01)
        ratio = np.maximum(125.0 * image_float / sum_channels, 0.01)
        color_restoration = np.log(ratio)

        # 最终增强结果
        enhanced = msr_result * color_restoration

        # 归一化到 [0, 255]
        enhanced = np.clip(enhanced, -3, 3)
        enhanced_range = enhanced.max() - enhanced.min()
        if enhanced_range > 0:
            enhanced = (enhanced - enhanced.min()) / enhanced_range
        else:
            enhanced = np.zeros_like(enhanced)
        enhanced = (enhanced * 255).astype(np.uint8)

        return enhanced

    except Exception as e:
        print(f"  ⚠ 图像增强失败: {e}，返回原图")
        return image


def sliding_window_predict(image, model, device, window_size=512, stride=256, threshold=0.5):
    """
    滑动窗口预测
    
    Args:
        image: 输入图像 (numpy array, BGR)
        model: 模型
        device: 设备
        window_size: 窗口大小
        stride: 步长
        threshold: 二值化阈值
    
    Returns:
        prediction: 预测结果 (numpy array, 0-255)
    """
    h, w = image.shape[:2]
    
    # 转换为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建预测结果和计数图
    prediction = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 滑动窗口
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 计算窗口边界
            y_end = min(y + window_size, h)
            x_end = min(x + window_size, w)
            
            # 提取窗口
            window = image_rgb[y:y_end, x:x_end]
            
            # 如果窗口小于标准大小，填充
            if window.shape[0] < window_size or window.shape[1] < window_size:
                padded = np.zeros((window_size, window_size, 3), dtype=np.uint8)
                padded[:window.shape[0], :window.shape[1]] = window
                window = padded
            
            # 转换为tensor
            window_pil = Image.fromarray(window)
            window_tensor = transform(window_pil).unsqueeze(0).to(device)
            
            # 预测
            with torch.no_grad():
                output = model(window_tensor)
                pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            
            # 裁剪到实际大小
            actual_h = min(window_size, y_end - y)
            actual_w = min(window_size, x_end - x)
            pred = pred[:actual_h, :actual_w]
            
            # 累加到结果
            prediction[y:y_end, x:x_end] += pred
            count_map[y:y_end, x:x_end] += 1
    
    # 平均
    count_map = np.maximum(count_map, 1)
    prediction = prediction / count_map
    
    # 二值化
    prediction = (prediction > threshold).astype(np.uint8) * 255
    
    return prediction


def process_image(image_path, model, device, args, output_dir):
    """
    处理单张图像

    Args:
        image_path: 图像路径
        model: 模型
        device: 设备
        args: 参数
        output_dir: 输出目录

    Returns:
        bool: 是否成功
    """
    try:
        # 读取图像
        image_original = cv2.imread(str(image_path))
        if image_original is None:
            print(f"  ❌ 无法读取图像: {image_path}")
            return False

        print(f"  图像尺寸: {image_original.shape[1]} × {image_original.shape[0]}")

        # 图像增强（如果启用）
        if args.enhance:
            print(f"  正在进行 Retinex 增强...")
            image_enhanced = retinex_enhancement(image_original)
            image_to_predict = image_enhanced
        else:
            image_to_predict = image_original

        # 滑动窗口预测
        print(f"  正在预测...")
        prediction = sliding_window_predict(
            image_to_predict, model, device,
            window_size=args.window_size,
            stride=args.stride,
            threshold=args.threshold
        )
        
        # 保存结果
        image_name = Path(image_path).stem

        # 1. 保存原图
        original_path = output_dir / f"{image_name}_original.jpg"
        cv2.imwrite(str(original_path), image_original)

        # 2. 如果启用了增强，保存增强后的图像
        if args.enhance:
            enhanced_path = output_dir / f"{image_name}_enhanced.jpg"
            cv2.imwrite(str(enhanced_path), image_enhanced)

        # 3. 保存预测掩码
        prediction_path = output_dir / f"{image_name}_prediction.png"
        cv2.imwrite(str(prediction_path), prediction)

        # 4. 保存可视化结果（叠加到原图上）
        overlay = image_original.copy()
        overlay[prediction > 0] = [0, 0, 255]  # 红色标记裂隙
        visualization = cv2.addWeighted(image_original, 0.7, overlay, 0.3, 0)
        visualization_path = output_dir / f"{image_name}_visualization.jpg"
        cv2.imwrite(str(visualization_path), visualization)

        print(f"  ✅ 处理完成")
        print(f"     - 原图: {original_path.name}")
        if args.enhance:
            print(f"     - 增强图: {enhanced_path.name}")
        print(f"     - 预测掩码: {prediction_path.name}")
        print(f"     - 可视化: {visualization_path.name}")
        
        return True

    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='简化版裂隙预测 - 批量处理子文件夹')
    parser.add_argument('--input_dir', type=str, default='data/cut_picture',
                       help='输入根目录（包含多个子文件夹）')
    parser.add_argument('--output_dir', type=str, default='data/prediction_results',
                       help='输出根目录')
    parser.add_argument('--model_path', type=str,
                       default='./checkpoints/weights/checkpoint_best.pth',
                       help='模型权重文件路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备: cuda 或 cpu')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值 (0-1)')
    parser.add_argument('--window_size', type=int, default=512, help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=256, help='滑动步长')
    parser.add_argument('--enhance', action='store_true',
                       help='启用 Retinex 图像增强（推荐）')
    parser.add_argument('--BCELoss_ratio', default=0.2, type=float)
    parser.add_argument('--DiceLoss_ratio', default=0.8, type=float)
    parser.add_argument('--Norm_Type', default='GN', type=str)

    args = parser.parse_args()

    # 检查输入目录
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ 错误: 输入目录不存在: {input_dir}")
        return 1

    # 检查模型文件
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        return 1

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ 警告: CUDA不可用，将使用CPU")

    # 加载模型
    print("\n正在加载模型...")
    try:
        model, _ = build_model(args)
        state_dict = torch.load(str(model_path), map_location=device)
        model.load_state_dict(state_dict["model"])
        model.to(device)
        model.eval()
        print("✓ 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 获取所有子文件夹
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if len(subdirs) == 0:
        print(f"❌ 错误: 在 {input_dir} 中未找到子文件夹")
        return 1

    print(f"\n找到 {len(subdirs)} 个子文件夹:")
    for subdir in subdirs:
        print(f"  - {subdir.name}")

    # 创建输出根目录
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # 统计信息
    total_images = 0
    total_success = 0
    total_failed = 0

    # 处理每个子文件夹
    for subdir in subdirs:
        print(f"\n{'='*80}")
        print(f"处理文件夹: {subdir.name}")
        print(f"{'='*80}")

        # 获取该文件夹中的所有图片
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG']
        image_paths = []
        for ext in supported_formats:
            image_paths.extend(list(subdir.glob(f'*{ext}')))

        if len(image_paths) == 0:
            print(f"  ⚠ 警告: 未找到图片文件，跳过")
            continue

        print(f"  找到 {len(image_paths)} 张图片")

        # 创建对应的输出文件夹
        output_subdir = output_root / subdir.name
        output_subdir.mkdir(parents=True, exist_ok=True)
        print(f"  输出目录: {output_subdir}")

        # 处理该文件夹中的每张图片
        success_count = 0
        failed_count = 0

        for i, image_path in enumerate(image_paths, 1):
            print(f"\n  [{i}/{len(image_paths)}] 处理: {image_path.name}")

            if process_image(image_path, model, device, args, output_subdir):
                success_count += 1
            else:
                failed_count += 1

        # 打印该文件夹的统计信息
        print(f"\n  文件夹 {subdir.name} 处理完成:")
        print(f"    ✓ 成功: {success_count} 张")
        print(f"    ❌ 失败: {failed_count} 张")

        total_images += len(image_paths)
        total_success += success_count
        total_failed += failed_count

    # 打印总体统计信息
    print(f"\n{'='*80}")
    print(f"批量处理完成!")
    print(f"{'='*80}")
    print(f"总图片数: {total_images}")
    print(f"✓ 成功处理: {total_success} 张")
    print(f"❌ 失败: {total_failed} 张")
    if total_images > 0:
        print(f"成功率: {total_success/total_images*100:.1f}%")
    print(f"{'='*80}")
    print(f"\n所有结果已保存到: {output_root}")

    return 0


if __name__ == '__main__':
    exit(main())

