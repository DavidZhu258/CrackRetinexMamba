import os.path
import cv2
from PIL import Image
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from .image_folder import make_dataset
from .utils import MaskToTensor

class FlexibleCrackDataset(BaseDataset):
    """灵活的裂缝数据集类，支持不同的数据集结构"""

    def __init__(self, args):
        """Initialize this dataset class.

        Parameters:
            args (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, args)
        
        # 根据test_dataset参数选择不同的数据路径
        if hasattr(args, 'test_dataset') and args.test_dataset == 'scsegamba':
            # 使用test_base作为图像，test_lab作为标签
            self.img_dir = os.path.join(args.dataset_path, 'test_base')
            self.lab_dir = os.path.join(args.dataset_path, 'test_lab')

            # 获取所有图像文件
            all_img_paths = make_dataset(self.img_dir)

            # 过滤出有对应标签的图像文件
            self.img_paths = []
            skipped_count = 0

            for img_path in all_img_paths:
                img_basename = os.path.basename(img_path)
                img_name_without_ext = os.path.splitext(img_basename)[0]
                lab_path = os.path.join(self.lab_dir, img_name_without_ext + '.png')

                if os.path.exists(lab_path):
                    self.img_paths.append(img_path)
                else:
                    skipped_count += 1

            print(f"使用SCSegamba数据集")
            print(f"图像目录: {self.img_dir} (原始图像)")
            print(f"标签目录: {self.lab_dir} (对应标注)")
            print(f"找到 {len(all_img_paths)} 个图像文件")
            print(f"有对应标签的图像: {len(self.img_paths)} 个")
            if skipped_count > 0:
                print(f"跳过无对应标签的图像: {skipped_count} 个")
        else:
            # 使用原始的TUT数据集结构
            self.img_paths = make_dataset(os.path.join(args.dataset_path, '{}_img'.format(args.phase)))
            self.lab_dir = os.path.join(args.dataset_path, '{}_lab'.format(args.phase))
            print(f"使用原始TUT数据集结构")
        
        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])
        self.lab_transform = MaskToTensor()
        self.phase = args.phase
        self.test_dataset = getattr(args, 'test_dataset', 'original')
        
        print(f"数据集初始化完成，共找到 {len(self.img_paths)} 张图片")

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            image (tensor) - - an image
            label (tensor) - - its corresponding segmentation
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        img_path = self.img_paths[index]
        
        # 根据不同的数据集类型处理标签路径
        if self.test_dataset == 'scsegamba':
            # 对于scsegamba数据集，标签文件名与图像文件名相同，但扩展名为.png
            img_basename = os.path.basename(img_path)
            img_name_without_ext = os.path.splitext(img_basename)[0]
            lab_path = os.path.join(self.lab_dir, img_name_without_ext + '.png')
        else:
            # 原始TUT数据集结构
            lab_path = os.path.join(self.lab_dir, os.path.basename(img_path).split('.')[0] + '.png')

        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"无法读取图像文件: {img_path}")
        
        # 如果是灰度图像，转换为RGB
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 读取标签
        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)
        if lab is None:
            print(f"错误详情:")
            print(f"  图像文件: {img_path}")
            print(f"  标签文件: {lab_path}")
            print(f"  标签文件存在: {os.path.exists(lab_path)}")
            raise ValueError(f"无法读取标签文件: {lab_path}")

        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

        # adjust the image size
        w, h = self.args.load_width, self.args.load_height
        if w > 0 or h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)

        # 二值化标签
        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)

        img = self.img_transforms(Image.fromarray(img.copy()))
        lab = self.lab_transform(lab.copy()).unsqueeze(0)
        
        return {'image': img, 'label': lab, 'A_paths': img_path, 'B_paths': lab_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)
