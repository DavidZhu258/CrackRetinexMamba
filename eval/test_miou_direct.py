#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/dministrator/app/cv_new/SCSegamba/eval')

# 直接导入函数
from evaluate import cal_mIoU_metrics as cal_mIoU_evaluate
from evaluate_1 import cal_mIoU_metrics as cal_mIoU_evaluate_1
from evaluate import get_image_pairs

# 加载数据
results_dir = "../results/results_test/2025_10_11_16:19:24_Dataset->MID_original"
src_img_list, tgt_img_list, pred_imgs_names, gt_imgs_names = get_image_pairs(results_dir, "lab", "pre")

print(f"加载了 {len(src_img_list)} 对图像")
print()

# 计算mIoU
print("计算 evaluate.py 的 mIoU...")
miou_evaluate = cal_mIoU_evaluate(src_img_list, tgt_img_list)
print(f"evaluate.py:   mIoU = {miou_evaluate:.4f} ({miou_evaluate*100:.2f}%)")
print()

print("计算 evaluate_1.py 的 mIoU...")
miou_evaluate_1 = cal_mIoU_evaluate_1(src_img_list, tgt_img_list)
print(f"evaluate_1.py: mIoU = {miou_evaluate_1:.4f} ({miou_evaluate_1*100:.2f}%)")
print()

if abs(miou_evaluate - miou_evaluate_1) < 0.0001:
    print("✅ 两个函数结果一致!")
else:
    print(f"❌ 两个函数结果不一致! 差异: {abs(miou_evaluate - miou_evaluate_1):.4f}")

