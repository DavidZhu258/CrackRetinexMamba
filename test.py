import numpy as np
import torch
import argparse
import os
import cv2
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import create_dataset
from models import build_model
from main import get_args_parser

def calculate_iou(pred, target, threshold=0.5):
    """计算IoU (Intersection over Union)"""
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > threshold).astype(np.uint8)

    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union

def calculate_dice(pred, target, threshold=0.5):
    """计算Dice系数"""
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > threshold).astype(np.uint8)

    intersection = np.logical_and(pred_binary, target_binary).sum()
    total = pred_binary.sum() + target_binary.sum()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2.0 * intersection / total

def calculate_metrics(pred, target, threshold=0.5):
    """计算所有评估指标"""
    pred_binary = (pred > threshold).astype(np.uint8).flatten()
    target_binary = (target > threshold).astype(np.uint8).flatten()

    accuracy = accuracy_score(target_binary, pred_binary)
    precision = precision_score(target_binary, pred_binary, zero_division=0)
    recall = recall_score(target_binary, pred_binary, zero_division=0)
    f1 = f1_score(target_binary, pred_binary, zero_division=0)
    iou = calculate_iou(pred, target, threshold)
    dice = calculate_dice(pred, target, threshold)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'dice': dice
    }

parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
parser.add_argument('--test_dataset', type=str, default='original',
                   choices=['original'],
                   help='选择测试数据集: original(原始TUT), scsegamba(SCSegamba数据集), crack_filter(crack_filter数据集), crack_full(整理后的crack_full数据集), crack_slide(滑动窗口数据集), crack_new(叠加数据集), crack_new_full(重建的完整叠加数据集), crack_final_slide_no_black(crack_res滑动窗口数据集), res_tut(RES/TUT合并数据集)')
parser.add_argument('--dataset_root', type=str, default='../data/MID',
                   help='数据集根目录路径')

args = parser.parse_args()
args.phase = 'test'

# 根据选择的测试数据集设置路径和数据集模式
if args.test_dataset == 'original':
    args.dataset_path = args.dataset_root
    args.dataset_mode = 'crack'  # 使用原始的CrackDataset
    print(f"使用原始TUT测试集: {args.dataset_path}")
    print("  - 图像来源: test_img目录")
    print("  - 标签来源: test_lab目录")

print(f"数据集路径: {args.dataset_path}")
print(f"数据集模式: {args.dataset_mode}")
print(f"测试集类型: {args.test_dataset}")

if __name__ == '__main__':
    args.batch_size = 1
    device = torch.device(args.device)
    test_dl = create_dataset(args)
    load_model_file = "./checkpoints/weights/2025_10_17_00:56:40_Dataset->MID/checkpoint_best.pth"
    data_size = len(test_dl)
    model, criterion = build_model(args)
    state_dict = torch.load(load_model_file)
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("Load Model Successful!")
    suffix = load_model_file.split('/')[-2]
    # 根据测试数据集类型创建不同的保存目录
    if args.test_dataset == 'original':
        save_root = f"./results/results_test/{suffix}_original"
    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    print(f"结果将保存到: {save_root}")

    # 初始化评估指标累积器
    all_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'iou': [],
        'dice': []
    }

    inference_times = []

    print(f"开始测试，共 {data_size} 张图片...")
    print("=" * 80)

    with torch.no_grad():
        model.eval()
        for batch_idx, (data) in enumerate(test_dl):
            x = data["image"]
            target = data["label"]
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()

            # 记录推理时间
            start_time = time.time()
            out = model(x)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            target_np = target[0, 0, ...].cpu().numpy()
            out_np = out[0, 0, ...].cpu().numpy()
            root_name = data["A_paths"][0].split("/")[-1][0:-4]

            # 归一化到0-1范围用于计算指标
            target_norm = target_np / np.max(target_np) if np.max(target_np) > 0 else target_np
            out_norm = out_np / np.max(out_np) if np.max(out_np) > 0 else out_np

            # 计算评估指标
            metrics = calculate_metrics(out_norm, target_norm, threshold=0.5)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])

            # 保存图片（归一化到0-255）
            target_save = 255 * target_norm
            out_save = 255 * out_norm

            cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target_save)
            cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out_save)

            # 打印当前图片的指标
            print(f"图片 {batch_idx+1}/{data_size}: {root_name}")
            print(f"  IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}, "
                  f"F1: {metrics['f1_score']:.4f}, Acc: {metrics['accuracy']:.4f}")
            print(f"  推理时间: {inference_time:.4f}s")
            print("-" * 60)

    # 计算平均指标
    print("\n" + "=" * 80)
    print("测试完成！总体评估结果:")
    print("=" * 80)

    for key in all_metrics:
        mean_val = np.mean(all_metrics[key])
        std_val = np.std(all_metrics[key])
        print(f"{key.upper():>12}: {mean_val:.4f} ± {std_val:.4f}")

    mean_inference_time = np.mean(inference_times)
    print(f"{'INFERENCE_TIME':>12}: {mean_inference_time:.4f}s ± {np.std(inference_times):.4f}s")
    print(f"{'FPS':>12}: {1.0/mean_inference_time:.2f}")

    # 保存结果到文件
    results_file = os.path.join(save_root, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("SCSegamba 测试结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"测试数据集类型: {args.test_dataset}\n")
        f.write(f"数据集路径: {args.dataset_path}\n")
        f.write(f"测试图片数量: {data_size}\n")
        f.write(f"模型文件: {load_model_file}\n")
        f.write(f"设备: {device}\n\n")

        f.write("评估指标:\n")
        for key in all_metrics:
            mean_val = np.mean(all_metrics[key])
            std_val = np.std(all_metrics[key])
            f.write(f"{key.upper():>12}: {mean_val:.4f} ± {std_val:.4f}\n")

        f.write(f"\n推理性能:\n")
        f.write(f"{'INFERENCE_TIME':>12}: {mean_inference_time:.4f}s ± {np.std(inference_times):.4f}s\n")
        f.write(f"{'FPS':>12}: {1.0/mean_inference_time:.2f}\n")

    print(f"\n详细结果已保存到: {results_file}")
    print("Finished!")
