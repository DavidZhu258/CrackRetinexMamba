#!/usr/bin/env python3
"""
创建Enhanced图像文件夹
enhance文件夹 - 纯净的Enhanced原图
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

class EnhanceFolderCreator:
    """Enhanced文件夹创建器"""
    
    def __init__(self,
                 input_dir: str = "ori-images\images",
                 output_base: str = "."):

        self.input_dir = Path(input_dir)
        self.output_base = Path(output_base)

        # 创建目标文件夹
        self.enhance_dir = self.output_base / "one-step-images"

        self.enhance_dir.mkdir(parents=True, exist_ok=True)

        print(f"🚀 初始化Enhanced文件夹创建器")
        print(f"📁 输入目录: {self.input_dir}")
        print(f"📁 Enhanced原图: {self.enhance_dir}")
    
    def cracknex_retinex_enhancement(self, image):
        """CrackNex Retinex增强"""
        image_float = image.astype(np.float32) / 255.0
        
        # 多尺度Retinex
        scales = [15, 80, 250]
        msr_result = np.zeros_like(image_float)
        
        for scale in scales:
            illumination = cv2.GaussianBlur(image_float, (0, 0), scale)
            illumination = np.maximum(illumination, 0.01)
            reflectance = np.log(image_float + 0.01) - np.log(illumination + 0.01)
            msr_result += reflectance
        
        msr_result = msr_result / len(scales)
        
        # 颜色恢复
        sum_channels = np.sum(image_float, axis=2, keepdims=True)
        sum_channels = np.maximum(sum_channels, 0.01)
        ratio = np.maximum(125.0 * image_float / sum_channels, 0.01)
        color_restoration = np.log(ratio)
        
        enhanced = msr_result * color_restoration
        
        # 归一化
        enhanced = np.clip(enhanced, -3, 3)
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
        enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced
    

    
    def process_image(self, image_path: Path):
        """处理单张图像"""
        print(f"🔍 处理图像: {image_path.name}")

        # 读取原图
        original = cv2.imread(str(image_path))
        if original is None:
            print(f"❌ 无法读取图像: {image_path}")
            return False

        base_name = image_path.stem

        # 生成Enhanced图像
        enhanced = self.cracknex_retinex_enhancement(original)

        # 保存纯净的Enhanced图像到enhance文件夹
        enhance_path = self.enhance_dir / f"{base_name}.jpg"
        cv2.imwrite(str(enhance_path), enhanced)
        print(f"  ✅ Enhanced原图: {enhance_path}")

        return True
    
    def process_batch(self, image_names=None):
        """批量处理"""
        if image_names:
            image_paths = [self.input_dir / f"{name}.jpg" for name in image_names]
            image_paths = [p for p in image_paths if p.exists()]
        else:
            image_paths = list(self.input_dir.glob("*.jpg"))
        
        if not image_paths:
            print("❌ 未找到要处理的图像")
            return
        
        print(f"📊 找到 {len(image_paths)} 张图像待处理")
        
        success_count = 0
        for image_path in image_paths:
            if self.process_image(image_path):
                success_count += 1
        
        print(f"\n🎉 处理完成！成功处理 {success_count}/{len(image_paths)} 张图像")
        print(f"📁 Enhanced原图: {self.enhance_dir}")
        
        return success_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="创建Enhanced图像文件夹")
    parser.add_argument("--input_dir", default="scaled_dataset/original_images", 
                       help="输入图像目录")
    parser.add_argument("--output_base", default=".", 
                       help="输出基础目录")
    parser.add_argument("--image", type=str, help="处理单张图像")
    parser.add_argument("--images", nargs="+", help="处理指定图像列表")
    parser.add_argument("--batch", action="store_true", help="批量处理")
    
    args = parser.parse_args()
    
    print("🎨 Enhanced文件夹创建系统")
    print("="*50)
    print("📁 enhance - Enhanced原图")
    print("="*50)
    
    creator = EnhanceFolderCreator(
        input_dir=args.input_dir,
        output_base=args.output_base
    )
    
    if args.image:
        image_names = [args.image]
        print(f"📷 处理模式: 单张图像 - {args.image}")
    elif args.images:
        image_names = args.images
        print(f"📷 处理模式: 指定图像 - {len(image_names)} 张")
    elif args.batch:
        image_names = None
        print(f"📷 处理模式: 批量处理")
    else:
        print("❌ 请指定处理模式: --image, --images, 或 --batch")
        return 1
    
    success_count = creator.process_batch(image_names)
    
    if success_count > 0:
        print(f"\n💡 使用说明:")
        print(f"enhance/ - 纯净的Enhanced图像，用于高质量显示和标注")
    
    return 0


if __name__ == "__main__":
    exit(main())
