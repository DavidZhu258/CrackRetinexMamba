import os.path
import cv2
import numpy as np
from PIL import Image
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from .image_folder import make_dataset
from .utils import MaskToTensor
import sys
import os

# 添加当前目录到路径，以便导入retinex_enhancement模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retinex_enhancement import RetinexEnhancer

class RetinexCrackDataset(BaseDataset):
    """
    集成Retinex增强的裂缝数据集类
    基于CrackNex论文的Retinex理论实现低光照图像增强
    """

    def __init__(self, args):
        """
        初始化数据集类
        
        Parameters:
            args (Option class) -- 存储所有实验参数，需要是BaseOptions的子类
        """
        BaseDataset.__init__(self, args)
        self.img_paths = make_dataset(os.path.join(args.dataset_path, '{}_img'.format(args.phase)))
        self.lab_dir = os.path.join(args.dataset_path, '{}_lab'.format(args.phase))
        
        # 图像预处理变换
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.lab_transform = MaskToTensor()
        
        self.phase = args.phase
        
        # 初始化Retinex增强器
        self.use_retinex = getattr(args, 'use_retinex', False)
        self.retinex_method = getattr(args, 'retinex_method', 'msrcr')
        self.retinex_adaptive = getattr(args, 'retinex_adaptive', True)
        self.low_light_threshold = getattr(args, 'low_light_threshold', 100.0)
        
        if self.use_retinex:
            # 创建Retinex增强器实例
            self.retinex_enhancer = RetinexEnhancer(
                sigma_list=[15, 80, 250],  # 多尺度参数
                G=5.0,
                b=25.0,
                alpha=125.0,
                beta=46.0,
                low_clip=0.01,
                high_clip=0.99
            )
            print(f"已启用Retinex增强 - 方法: {self.retinex_method}, 自适应: {self.retinex_adaptive}")
        else:
            self.retinex_enhancer = None

    def _apply_retinex_enhancement(self, img):
        """
        应用Retinex增强
        
        Args:
            img: 输入图像 (numpy array, BGR格式)
            
        Returns:
            增强后的图像
        """
        if not self.use_retinex or self.retinex_enhancer is None:
            return img
            
        try:
            if self.retinex_adaptive:
                # 自适应增强：根据图像亮度决定是否增强
                enhanced_img = self.retinex_enhancer.adaptive_enhance(img)
            else:
                # 强制增强：总是应用Retinex增强
                enhanced_img = self.retinex_enhancer.enhance_image(img, method=self.retinex_method)
                
            return enhanced_img
            
        except Exception as e:
            print(f"Retinex增强失败: {str(e)}")
            return img  # 增强失败时返回原图

    def _is_low_light_image(self, img):
        """
        判断是否为低光照图像
        
        Args:
            img: 输入图像 (numpy array)
            
        Returns:
            bool: 是否为低光照图像
        """
        if len(img.shape) == 3:
            # 转换为灰度图计算平均亮度
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        mean_brightness = np.mean(gray)
        return mean_brightness < self.low_light_threshold

    def __getitem__(self, index):
        """
        返回数据点及其元数据信息
        
        Parameters:
            index -- 数据索引的随机整数
            
        Returns:
            包含以下内容的字典：
                image (tensor) -- 图像
                label (tensor) -- 对应的分割标签
                A_paths (str) -- 图像路径
                B_paths (str) -- 标签路径
                is_enhanced (bool) -- 是否应用了Retinex增强
                original_brightness (float) -- 原始图像亮度
        """
        # 读取图像
        img_path = self.img_paths[index]
        lab_path = os.path.join(self.lab_dir, os.path.basename(img_path).split('.')[0] + '.png')

        # 读取原始图像
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
            
        # 记录原始亮度
        original_brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        
        # 应用Retinex增强
        is_enhanced = False
        if self.use_retinex:
            if self.retinex_adaptive:
                # 自适应增强：只对低光照图像增强
                if self._is_low_light_image(img):
                    img = self._apply_retinex_enhancement(img)
                    is_enhanced = True
            else:
                # 强制增强：对所有图像增强
                img = self._apply_retinex_enhancement(img)
                is_enhanced = True
        
        # 转换色彩空间
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 读取标签
        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)
        if lab is None:
            raise ValueError(f"无法读取标签: {lab_path}")

        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

        # 调整图像尺寸
        w, h = self.args.load_width, self.args.load_height
        if w > 0 or h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)

        # 标签二值化
        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)

        # 转换为张量
        img = self.img_transforms(Image.fromarray(img.copy()))
        lab = self.lab_transform(lab.copy()).unsqueeze(0)
        
        return {
            'image': img, 
            'label': lab, 
            'A_paths': img_path, 
            'B_paths': lab_path,
            'is_enhanced': is_enhanced,
            'original_brightness': original_brightness
        }

    def __len__(self):
        """返回数据集中图像的总数"""
        return len(self.img_paths)

    def get_enhancement_stats(self):
        """
        获取增强统计信息
        
        Returns:
            dict: 包含增强统计信息的字典
        """
        if not self.use_retinex:
            return {"retinex_enabled": False}
            
        stats = {
            "retinex_enabled": True,
            "enhancement_method": self.retinex_method,
            "adaptive_enhancement": self.retinex_adaptive,
            "low_light_threshold": self.low_light_threshold,
            "total_images": len(self.img_paths)
        }
        
        if self.retinex_adaptive:
            # 统计低光照图像数量
            low_light_count = 0
            for img_path in self.img_paths[:min(100, len(self.img_paths))]:  # 采样前100张图像
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is not None and self._is_low_light_image(img):
                    low_light_count += 1
                    
            stats["estimated_low_light_ratio"] = low_light_count / min(100, len(self.img_paths))
            
        return stats


class RetinexFlexibleCrackDataset(BaseDataset):
    """
    灵活的Retinex裂缝数据集类
    支持更多的数据组织格式和增强选项
    """

    def __init__(self, args):
        """初始化灵活的Retinex数据集"""
        BaseDataset.__init__(self, args)
        
        # 支持多种数据组织格式
        self.data_format = getattr(args, 'data_format', 'standard')  # 'standard', 'mixed', 'custom'
        
        if self.data_format == 'standard':
            # 标准格式: train_img, train_lab, test_img, test_lab
            self.img_paths = make_dataset(os.path.join(args.dataset_path, '{}_img'.format(args.phase)))
            self.lab_dir = os.path.join(args.dataset_path, '{}_lab'.format(args.phase))
        elif self.data_format == 'mixed':
            # 混合格式: 图像和标签在同一目录
            self.img_paths = []
            self.lab_paths = []
            data_dir = os.path.join(args.dataset_path, args.phase)
            for file in os.listdir(data_dir):
                if file.endswith(('_img.jpg', '_img.png')):
                    img_path = os.path.join(data_dir, file)
                    lab_path = os.path.join(data_dir, file.replace('_img', '_lab'))
                    if os.path.exists(lab_path):
                        self.img_paths.append(img_path)
                        self.lab_paths.append(lab_path)
        
        # 图像预处理
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.lab_transform = MaskToTensor()
        
        # Retinex增强配置
        self.use_retinex = getattr(args, 'use_retinex', False)
        self.retinex_config = {
            'method': getattr(args, 'retinex_method', 'msrcr'),
            'adaptive': getattr(args, 'retinex_adaptive', True),
            'threshold': getattr(args, 'low_light_threshold', 100.0),
            'sigma_list': getattr(args, 'retinex_sigma_list', [15, 80, 250]),
            'G': getattr(args, 'retinex_G', 5.0),
            'b': getattr(args, 'retinex_b', 25.0),
            'alpha': getattr(args, 'retinex_alpha', 125.0),
            'beta': getattr(args, 'retinex_beta', 46.0),
        }
        
        if self.use_retinex:
            self.retinex_enhancer = RetinexEnhancer(
                sigma_list=self.retinex_config['sigma_list'],
                G=self.retinex_config['G'],
                b=self.retinex_config['b'],
                alpha=self.retinex_config['alpha'],
                beta=self.retinex_config['beta']
            )
        else:
            self.retinex_enhancer = None
            
        self.phase = args.phase

    def __getitem__(self, index):
        """获取数据项"""
        # 获取图像和标签路径
        img_path = self.img_paths[index]
        
        if self.data_format == 'standard':
            lab_path = os.path.join(self.lab_dir, os.path.basename(img_path).split('.')[0] + '.png')
        else:  # mixed or custom
            lab_path = self.lab_paths[index]

        # 读取和处理图像
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        original_brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        
        # 应用Retinex增强
        is_enhanced = False
        if self.use_retinex and self.retinex_enhancer:
            if self.retinex_config['adaptive']:
                if original_brightness < self.retinex_config['threshold']:
                    img = self.retinex_enhancer.enhance_image(img, method=self.retinex_config['method'])
                    is_enhanced = True
            else:
                img = self.retinex_enhancer.enhance_image(img, method=self.retinex_config['method'])
                is_enhanced = True
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 读取和处理标签
        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)
        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

        # 尺寸调整
        w, h = self.args.load_width, self.args.load_height
        if w > 0 or h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)

        # 标签二值化
        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)

        # 转换为张量
        img = self.img_transforms(Image.fromarray(img.copy()))
        lab = self.lab_transform(lab.copy()).unsqueeze(0)
        
        return {
            'image': img, 
            'label': lab, 
            'A_paths': img_path, 
            'B_paths': lab_path,
            'is_enhanced': is_enhanced,
            'original_brightness': original_brightness,
            'enhancement_config': self.retinex_config if is_enhanced else None
        }

    def __len__(self):
        """返回数据集大小"""
        return len(self.img_paths)
