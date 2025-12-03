import numpy as np
import torch
import argparse
import os
import cv2
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from models import build_model
from main import get_args_parser
from scipy import ndimage
from skimage import morphology, measure
from scipy.interpolate import interp1d
from pathlib import Path

# 全局变量用于交互式区域选择
roi_points = []
roi_selecting = False
roi_image = None

def cracknex_retinex_enhancement(image):
    """
    CrackNex Retinex图像增强

    Args:
        image: OpenCV图像 (BGR格式)

    Returns:
        enhanced: 增强后的图像 (BGR格式)，如果失败返回原图
    """
    try:
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

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
        enhanced_range = enhanced.max() - enhanced.min()
        if enhanced_range > 0:
            enhanced = (enhanced - enhanced.min()) / enhanced_range
        else:
            enhanced = np.zeros_like(enhanced)
        enhanced = (enhanced * 255).astype(np.uint8)

        return enhanced
    except Exception as e:
        print(f"⚠ 图像增强失败: {e}，返回原图")
        return image

def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """
    在OpenCV图像上绘制中文文本

    Args:
        img: OpenCV图像 (BGR格式)
        text: 要绘制的文本
        position: 文本位置 (x, y)
        font_size: 字体大小
        color: 文本颜色 (BGR格式)

    Returns:
        img: 绘制后的图像
    """
    # 转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 尝试使用系统中文字体
    try:
        # Linux常见中文字体路径
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
            '/usr/share/fonts/truetype/arphic/uming.ttc',
            '/System/Library/Fonts/PingFang.ttc',  # macOS
            'C:\\Windows\\Fonts\\simhei.ttf',  # Windows
        ]

        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break

        if font is None:
            # 如果没有找到中文字体，使用默认字体
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # 绘制文本 (PIL使用RGB颜色)
    color_rgb = (color[2], color[1], color[0])  # BGR转RGB
    draw.text(position, text, font=font, fill=color_rgb)

    # 转换回OpenCV格式
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return img

def select_roi_interactive(image, window_name="选择感兴趣区域 - 左键点击绘制多边形，右键完成，ESC取消"):
    """
    交互式选择感兴趣区域（ROI）

    Args:
        image: numpy数组 (RGB格式)
        window_name: 窗口名称

    Returns:
        roi_mask: 二值mask，选中区域为255，其他为0
        roi_points: 多边形顶点列表
    """
    global roi_points, roi_selecting, roi_image

    roi_points = []
    roi_selecting = True

    # 转换为BGR用于显示
    if len(image.shape) == 2:
        roi_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        roi_image = image.copy()

    display_image = roi_image.copy()

    def mouse_callback(event, x, y, flags, param):
        global roi_points, roi_selecting, roi_image

        if not roi_selecting:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键点击添加点
            roi_points.append((x, y))
            # 重绘图像
            temp_img = roi_image.copy()

            # 绘制已有的点
            for i, pt in enumerate(roi_points):
                cv2.circle(temp_img, pt, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(temp_img, roi_points[i-1], pt, (0, 255, 0), 2)

            # 如果有多个点，绘制临时闭合线
            if len(roi_points) > 2:
                cv2.line(temp_img, roi_points[-1], roi_points[0], (0, 255, 0), 1)

            cv2.imshow(window_name, temp_img)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键完成选择
            roi_selecting = False
            cv2.destroyAllWindows()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)
    cv2.imshow(window_name, display_image)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "=" * 80)
    print("交互式区域选择:")
    print("=" * 80)
    print("  - 左键点击: 添加多边形顶点")
    print("  - 右键点击: 完成选择")
    print("  - ESC键: 取消选择（使用全图）")
    print("=" * 80)

    while roi_selecting:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            roi_points = []
            roi_selecting = False
            cv2.destroyAllWindows()
            print("✓ 已取消区域选择，将使用全图")
            return None, []

    cv2.destroyAllWindows()

    if len(roi_points) < 3:
        print("✓ 选择的点少于3个，将使用全图")
        return None, []

    # 创建mask
    mask = np.zeros(roi_image.shape[:2], dtype=np.uint8)
    pts = np.array(roi_points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    print(f"✓ 已选择区域，包含 {len(roi_points)} 个顶点")

    return mask, roi_points

def filter_cracks_by_roi(crack_info, skeleton, roi_mask):
    """
    根据ROI mask过滤裂缝

    Args:
        crack_info: 裂缝信息列表
        skeleton: 骨架图
        roi_mask: ROI mask

    Returns:
        filtered_crack_info: 过滤后的裂缝信息列表
    """
    if roi_mask is None:
        return crack_info

    filtered_cracks = []

    for crack in crack_info:
        # 获取裂缝的中心点
        cy, cx = crack['centroid']
        cy, cx = int(cy), int(cx)

        # 检查中心点是否在ROI内
        if cy < roi_mask.shape[0] and cx < roi_mask.shape[1]:
            if roi_mask[cy, cx] > 0:
                filtered_cracks.append(crack)

    return filtered_cracks

def detect_valid_region(image, threshold=10):
    """
    检测图像中的有效区域（非黑色/背景区域）

    Args:
        image: PIL Image对象或numpy数组
        threshold: 像素值阈值，低于此值认为是背景（默认10）

    Returns:
        valid_mask: 有效区域的二值掩码（numpy数组）
        valid_area_pixels: 有效区域的像素数量
    """
    # 转换为numpy数组
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image

    # 转换为灰度图
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    # 创建有效区域掩码（像素值大于阈值的区域）
    valid_mask = (gray > threshold).astype(np.uint8)

    # 形态学操作去除噪点
    kernel = np.ones((5, 5), np.uint8)
    valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel)
    valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_OPEN, kernel)

    # 计算有效区域的像素数量
    valid_area_pixels = np.sum(valid_mask)

    return valid_mask, valid_area_pixels


def calculate_line_crack_intersections(line_start, line_end, skeleton):
    """
    计算测线与裂缝骨架的交点

    Args:
        line_start: 测线起点 (x, y)
        line_end: 测线终点 (x, y)
        skeleton: 裂缝骨架图 (二值图)

    Returns:
        intersections: 交点列表，每个元素为 (x, y, distance_from_start)
    """
    x1, y1 = line_start
    x2, y2 = line_end

    # 计算测线长度
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # 生成测线上的采样点（每个像素采样）
    num_samples = int(line_length) + 1
    if num_samples < 2:
        return []

    x_samples = np.linspace(x1, x2, num_samples)
    y_samples = np.linspace(y1, y2, num_samples)

    intersections = []
    prev_on_crack = False

    for i in range(num_samples):
        x, y = int(round(x_samples[i])), int(round(y_samples[i]))

        # 检查是否在图像范围内
        if 0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1]:
            on_crack = skeleton[y, x] > 0

            # 检测从非裂缝到裂缝的转换（交点）
            if on_crack and not prev_on_crack:
                distance = np.sqrt((x - x1)**2 + (y - y1)**2)
                intersections.append((x, y, distance))

            prev_on_crack = on_crack

    return intersections

def calculate_rqd_for_line(line_start, line_end, skeleton, pixel_to_cm, threshold_cm=20):
    """
    计算单条测线的RQD值和节理间距

    RQD公式：RQD = (交点间距≥20cm的段长度之和) / (第一个交点到最后一个交点的距离) × 100%
    节理间距公式：节理间距 = (第一个交点到最后一个交点的距离) / 交点个数

    Args:
        line_start: 测线起点 (x, y)
        line_end: 测线终点 (x, y)
        skeleton: 裂缝骨架图
        pixel_to_cm: 像素到厘米的转换比例
        threshold_cm: RQD阈值（默认20cm）

    Returns:
        rqd: RQD值 (0-100%)
        effective_length_cm: 有效测线长度（第一个交点到最后一个交点的距离）
        joint_spacing_cm: 节理间距（cm）
        segments: 段信息列表 [(start_dist, end_dist, length_cm, is_valid)]
        num_intersections: 交点数量
        intersections: 交点列表 [(x, y, distance_from_start)]
    """
    x1, y1 = line_start
    x2, y2 = line_end

    # 计算测线总长度（像素）
    line_length_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    line_length_cm = line_length_px * pixel_to_cm

    # 获取交点
    intersections = calculate_line_crack_intersections(line_start, line_end, skeleton)

    if len(intersections) == 0:
        # 没有裂缝交点，RQD无法计算，返回100%，节理间距为0
        return 100.0, line_length_cm, 0.0, [], 0, []

    if len(intersections) == 1:
        # 只有1个交点，无法形成段，RQD无法计算，节理间距为0
        return 0.0, 0.0, 0.0, [], 1, intersections

    # 计算交点间的段
    segments = []

    # 只计算交点之间的段（不包括起点到第一个交点、最后一个交点到终点）
    for i in range(len(intersections) - 1):
        start_dist_cm = intersections[i][2] * pixel_to_cm
        end_dist_cm = intersections[i + 1][2] * pixel_to_cm
        segment_length_cm = end_dist_cm - start_dist_cm

        if segment_length_cm > 0:
            is_valid = segment_length_cm >= threshold_cm
            segments.append((start_dist_cm, end_dist_cm, segment_length_cm, is_valid))

    # 计算有效测线长度：第一个交点到最后一个交点的距离
    first_intersection_cm = intersections[0][2] * pixel_to_cm
    last_intersection_cm = intersections[-1][2] * pixel_to_cm
    effective_length_cm = last_intersection_cm - first_intersection_cm

    # 计算RQD值：有效段长度之和 / 有效测线长度
    valid_length_cm = sum(seg[2] for seg in segments if seg[3])
    rqd = (valid_length_cm / effective_length_cm * 100) if effective_length_cm > 0 else 0

    # 计算节理间距：有效测线长度 / 交点个数
    joint_spacing_cm = effective_length_cm / len(intersections) if len(intersections) > 0 else 0

    return rqd, effective_length_cm, joint_spacing_cm, segments, len(intersections), intersections

def calculate_rqd_four_lines(image_shape, skeleton, pixel_to_cm, threshold_cm=20, margin_ratio=0.0):
    """
    在图像上绘制4条测线并计算RQD值和节理间距（2条横线 + 2条竖线，形成井字形）

    Args:
        image_shape: 图像形状 (height, width)
        skeleton: 裂缝骨架图
        pixel_to_cm: 像素到厘米的转换比例
        threshold_cm: RQD阈值（默认20cm）
        margin_ratio: 边距比例（默认0.0，即从边缘到边缘）

    Returns:
        rqd_avg: 平均RQD值
        joint_spacing_avg: 平均节理间距
        rqd_results: 每条测线的详细结果
        test_lines: 测线坐标列表 [(start, end), ...]
    """
    height, width = image_shape

    # 计算边距（默认为0，测线从边缘到边缘）
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)

    test_lines = []

    # 定义2条横向测线（水平线）- 从左边缘到右边缘
    # 将图像高度分为3等份，在1/3和2/3位置绘制横线
    h_positions = [height // 3, height * 2 // 3]
    for y in h_positions:
        start = (margin_x, y)
        end = (width - margin_x - 1, y)  # -1 确保不超出图像边界
        test_lines.append((start, end))

    # 定义2条纵向测线（垂直线）- 从上边缘到下边缘
    # 将图像宽度分为3等份，在1/3和2/3位置绘制竖线
    v_positions = [width // 3, width * 2 // 3]
    for x in v_positions:
        start = (x, margin_y)
        end = (x, height - margin_y - 1)  # -1 确保不超出图像边界
        test_lines.append((start, end))

    # 计算每条测线的RQD和节理间距
    rqd_results = []
    line_names = ['横线1', '横线2', '竖线1', '竖线2']

    for i, (start, end) in enumerate(test_lines):
        rqd, effective_length_cm, joint_spacing_cm, segments, num_intersections, intersections = calculate_rqd_for_line(
            start, end, skeleton, pixel_to_cm, threshold_cm
        )
        rqd_results.append({
            'line_id': i + 1,
            'line_name': line_names[i],
            'start': start,
            'end': end,
            'rqd': rqd,
            'effective_length_cm': effective_length_cm,  # 第一个交点到最后一个交点的距离
            'joint_spacing_cm': joint_spacing_cm,  # 节理间距
            'segments': segments,
            'num_intersections': num_intersections,
            'intersections': intersections  # 交点坐标列表
        })

    # 计算平均RQD
    rqd_avg = np.mean([r['rqd'] for r in rqd_results])

    # 计算平均节理间距
    joint_spacing_avg = np.mean([r['joint_spacing_cm'] for r in rqd_results])

    return rqd_avg, joint_spacing_avg, rqd_results, test_lines

def calculate_joint_spacing_four_lines(image_shape, skeleton, pixel_to_cm, margin_ratio=0.0):
    """
    在图像上绘制4条测线并计算节理间距（2条横线 + 2条竖线，形成井字形）

    Args:
        image_shape: 图像形状 (height, width)
        skeleton: 裂缝骨架图
        pixel_to_cm: 像素到厘米的转换比例
        margin_ratio: 边距比例（默认0.0，即从边缘到边缘）

    Returns:
        joint_spacing_avg: 平均节理间距
        results: 每条测线的详细结果
        test_lines: 测线坐标列表 [(start, end), ...]
    """
    height, width = image_shape

    # 计算边距（默认为0，测线从边缘到边缘）
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)

    test_lines = []

    # 定义2条横向测线（水平线）- 从左边缘到右边缘
    h_positions = [height // 3, height * 2 // 3]
    for y in h_positions:
        start = (margin_x, y)
        end = (width - margin_x - 1, y)
        test_lines.append((start, end))

    # 定义2条纵向测线（垂直线）- 从上边缘到下边缘
    v_positions = [width // 3, width * 2 // 3]
    for x in v_positions:
        start = (x, margin_y)
        end = (x, height - margin_y - 1)
        test_lines.append((start, end))

    # 计算每条测线的节理间距
    results = []
    line_names = ['横线1', '横线2', '竖线1', '竖线2']

    for i, (start, end) in enumerate(test_lines):
        # 检测交点（返回像素距离）
        intersections_px = calculate_line_crack_intersections(start, end, skeleton)

        # 将交点距离转换为厘米
        intersections = [(x, y, dist * pixel_to_cm) for x, y, dist in intersections_px]

        if len(intersections) >= 2:
            # 计算第一个交点到最后一个交点的距离（已经是厘米）
            first_dist = intersections[0][2]
            last_dist = intersections[-1][2]
            effective_length_cm = last_dist - first_dist

            # 节理间距 = 有效长度 / 交点个数
            joint_spacing_cm = effective_length_cm / len(intersections)
        else:
            effective_length_cm = 0
            joint_spacing_cm = 0

        results.append({
            'line_id': i + 1,
            'line_name': line_names[i],
            'start': start,
            'end': end,
            'effective_length_cm': effective_length_cm,
            'joint_spacing_cm': joint_spacing_cm,
            'num_intersections': len(intersections),
            'intersections': intersections
        })

    # 计算平均节理间距
    joint_spacing_avg = np.mean([r['joint_spacing_cm'] for r in results])

    return joint_spacing_avg, results, test_lines

def visualize_joint_spacing(skeleton, results, test_lines, pixel_to_cm):
    """
    在骨架图上可视化节理间距计算过程

    Args:
        skeleton: 裂缝骨架图（二值图）
        results: 节理间距计算结果列表
        test_lines: 测线坐标列表
        pixel_to_cm: 像素到厘米的转换比例

    Returns:
        vis_img: 可视化图像（BGR格式）
    """
    # 创建彩色图像
    if len(skeleton.shape) == 2:
        vis_img = cv2.cvtColor((skeleton * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        vis_img = skeleton.copy()

    # 定义颜色
    line_colors = [
        (0, 255, 255),    # H-Line1 - 黄色
        (0, 165, 255),    # H-Line2 - 橙色
        (255, 0, 255),    # V-Line1 - 品红
        (255, 255, 0),    # V-Line2 - 青色
    ]

    # 英文标签
    line_labels = ['H-Line1', 'H-Line2', 'V-Line1', 'V-Line2']

    # 绘制每条测线及其交点
    for i, (result, (line_start, line_end)) in enumerate(zip(results, test_lines)):
        color = line_colors[i]
        line_label = line_labels[i]
        intersections = result['intersections']

        # 1. 绘制测线（实线，从边缘到边缘）
        cv2.line(vis_img, line_start, line_end, color, thickness=3)

        # 2. 标记交点
        for j, (x, y, dist) in enumerate(intersections):
            # 绘制交点圆圈
            cv2.circle(vis_img, (int(x), int(y)), 6, (0, 0, 255), -1)  # 红色实心圆
            cv2.circle(vis_img, (int(x), int(y)), 8, (255, 255, 255), 2)  # 白色边框

            # 标注交点编号
            label = f"{chr(65+j)}"  # A, B, C, D, ...
            cv2.putText(vis_img, label, (int(x)+12, int(y)-12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 3. 在测线中点标注节理间距
            mid_x = (line_start[0] + line_end[0]) // 2
            mid_y = (line_start[1] + line_end[1]) // 2

            spacing_label = f"{line_label}: {result['joint_spacing_cm']:.1f}cm"

            # 根据测线类型调整标签位置
            if i < 2:  # 横线
                label_pos = (mid_x - 70, mid_y - 40 if i == 0 else mid_y + 50)
            else:  # 竖线
                label_pos = (mid_x - 90 if i == 2 else mid_x + 20, mid_y)

            # 绘制背景框
            (text_w, text_h), _ = cv2.getTextSize(spacing_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_img,
                         (label_pos[0]-5, label_pos[1]-text_h-5),
                         (label_pos[0]+text_w+5, label_pos[1]+5),
                         (0, 0, 0), -1)

            # 绘制文本
            cv2.putText(vis_img, spacing_label, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 4. 添加节理间距信息面板（右上角）
    joint_spacing_avg = np.mean([r['joint_spacing_cm'] for r in results])

    panel_x = vis_img.shape[1] - 320
    panel_y = 30
    panel_w = 300
    panel_h = 150

    # 绘制半透明背景
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, vis_img, 0.3, 0, vis_img)

    # 绘制边框
    cv2.rectangle(vis_img, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), (255, 255, 255), 2)

    # 标题
    cv2.putText(vis_img, "Joint Spacing Analysis", (panel_x+10, panel_y+25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # 各测线节理间距
    y_offset = 55
    for i, result in enumerate(results):
        text = f"{line_labels[i]}: {result['joint_spacing_cm']:.1f}cm"
        cv2.putText(vis_img, text, (panel_x+10, panel_y+y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_colors[i], 1)
        y_offset += 22

    # 平均节理间距
    cv2.putText(vis_img, f"Avg Spacing: {joint_spacing_avg:.1f}cm", (panel_x+10, panel_y+y_offset+5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return vis_img

def detect_ruler(image, ruler_length_cm=50):
    """
    检测图片中的黄色或红色尺子，并计算像素到厘米的比例

    Args:
        image: PIL Image对象或numpy数组
        ruler_length_cm: 尺子的实际长度(厘米)

    Returns:
        pixel_to_cm: 像素到厘米的比例
        ruler_mask: 尺子的mask
        ruler_bbox: 尺子的边界框 (x, y, w, h)
    """
    try:
        # 转换为numpy数组
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image.copy()

        if img_np is None or img_np.size == 0:
            raise ValueError("输入图像为空")

        # 转换到HSV色彩空间
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # 黄色范围 (HSV)
        yellow_lower1 = np.array([20, 100, 100])
        yellow_upper1 = np.array([30, 255, 255])
        yellow_mask1 = cv2.inRange(img_hsv, yellow_lower1, yellow_upper1)

        yellow_lower2 = np.array([15, 80, 80])
        yellow_upper2 = np.array([35, 255, 255])
        yellow_mask2 = cv2.inRange(img_hsv, yellow_lower2, yellow_upper2)

        # 红色范围 (HSV) - 红色在HSV中分两段
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_mask1 = cv2.inRange(img_hsv, red_lower1, red_upper1)

        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_mask2 = cv2.inRange(img_hsv, red_lower2, red_upper2)

        # 合并所有mask
        ruler_mask = cv2.bitwise_or(yellow_mask1, yellow_mask2)
        ruler_mask = cv2.bitwise_or(ruler_mask, red_mask1)
        ruler_mask = cv2.bitwise_or(ruler_mask, red_mask2)

        # 形态学操作，去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        ruler_mask = cv2.morphologyEx(ruler_mask, cv2.MORPH_CLOSE, kernel)
        ruler_mask = cv2.morphologyEx(ruler_mask, cv2.MORPH_OPEN, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(ruler_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            print("⚠ 警告: 未检测到尺子!")
            return None, ruler_mask, None

        # 找到最大的轮廓（假设是尺子）
        largest_contour = max(contours, key=cv2.contourArea)

        # 检查轮廓面积是否足够大
        contour_area = cv2.contourArea(largest_contour)
        if contour_area < 100:  # 最小面积阈值
            print(f"⚠ 警告: 检测到的尺子区域太小 (面积: {contour_area})")
            return None, ruler_mask, None

        x, y, w, h = cv2.boundingRect(largest_contour)

        # 计算尺子的长度（像素）
        # 假设尺子是水平或垂直放置的
        ruler_length_pixels = max(w, h)

        if ruler_length_pixels == 0:
            print("⚠ 警告: 尺子长度为0")
            return None, ruler_mask, None

        # 计算像素到厘米的比例
        pixel_to_cm = ruler_length_cm / ruler_length_pixels

        print(f"✓ 检测到尺子:")
        print(f"  位置: ({x}, {y})")
        print(f"  尺寸: {w}x{h} 像素")
        print(f"  长度: {ruler_length_pixels} 像素 = {ruler_length_cm} cm")
        print(f"  比例: 1 像素 = {pixel_to_cm:.4f} cm")

        return pixel_to_cm, ruler_mask, (x, y, w, h)

    except Exception as e:
        print(f"⚠ 尺子检测失败: {e}")
        # 返回默认的空mask
        if isinstance(image, Image.Image):
            h, w = image.size[1], image.size[0]
        else:
            h, w = image.shape[:2]
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        return None, empty_mask, None

def pad_image_to_multiple(image, multiple=512, fill_color='edge_mean'):
    """
    将图像填充到指定倍数的尺寸

    Args:
        image: PIL图像
        multiple: 倍数（默认512）
        fill_color: 填充颜色策略
            - 'edge_mean': 使用图像边缘的平均颜色（默认，推荐）
            - 'white': 白色填充
            - 'gray': 灰色填充
            - (R, G, B): 自定义颜色元组

    Returns:
        padded_image: 填充后的图像
        padding_info: 填充信息 (left, top, right, bottom)
    """
    width, height = image.size

    # 计算需要填充到的尺寸
    new_width = ((width + multiple - 1) // multiple) * multiple
    new_height = ((height + multiple - 1) // multiple) * multiple

    # 如果已经是倍数，不需要填充
    if new_width == width and new_height == height:
        return image, (0, 0, 0, 0)

    # 计算填充量（居中填充）
    pad_left = (new_width - width) // 2
    pad_top = (new_height - height) // 2
    pad_right = new_width - width - pad_left
    pad_bottom = new_height - height - pad_top

    # 确定填充颜色
    if fill_color == 'edge_mean':
        # 计算图像边缘的平均颜色
        img_np = np.array(image)

        # 提取边缘像素（上下左右各10像素）
        edge_pixels = []
        edge_width = min(10, width // 10, height // 10)

        # 上边缘
        edge_pixels.append(img_np[:edge_width, :, :].reshape(-1, 3))
        # 下边缘
        edge_pixels.append(img_np[-edge_width:, :, :].reshape(-1, 3))
        # 左边缘
        edge_pixels.append(img_np[:, :edge_width, :].reshape(-1, 3))
        # 右边缘
        edge_pixels.append(img_np[:, -edge_width:, :].reshape(-1, 3))

        # 合并所有边缘像素
        all_edge_pixels = np.vstack(edge_pixels)

        # 计算平均颜色
        mean_color = tuple(int(x) for x in np.mean(all_edge_pixels, axis=0))
        print(f"  边缘平均颜色: RGB{mean_color}")
    elif fill_color == 'white':
        mean_color = (255, 255, 255)
    elif fill_color == 'gray':
        mean_color = (128, 128, 128)
    else:
        mean_color = fill_color

    # 创建新图像（使用计算的填充颜色）
    padded_image = Image.new('RGB', (new_width, new_height), mean_color)
    padded_image.paste(image, (pad_left, pad_top))

    print(f"  图像填充: {width}x{height} -> {new_width}x{new_height}")
    print(f"  填充量: 左={pad_left}, 上={pad_top}, 右={pad_right}, 下={pad_bottom}")

    return padded_image, (pad_left, pad_top, pad_right, pad_bottom)


def create_padding_mask(padded_size, padding_info, original_size):
    """
    创建填充区域的掩码（填充区域=0，原始区域=1）

    Args:
        padded_size: 填充后的尺寸 (width, height)
        padding_info: 填充信息 (left, top, right, bottom)
        original_size: 原始图像尺寸 (width, height)

    Returns:
        mask: 填充掩码，numpy数组 (height, width)
    """
    pad_left, pad_top, pad_right, pad_bottom = padding_info
    original_width, original_height = original_size
    padded_width, padded_height = padded_size

    # 创建全0掩码
    mask = np.zeros((padded_height, padded_width), dtype=np.uint8)

    # 原始图像区域设为1
    mask[pad_top:pad_top+original_height, pad_left:pad_left+original_width] = 1

    return mask


def unpad_prediction(prediction, padding_info, original_size):
    """
    移除预测结果中的填充部分

    Args:
        prediction: 填充后的预测结果
        padding_info: 填充信息 (left, top, right, bottom)
        original_size: 原始图像尺寸 (width, height)

    Returns:
        unpadded_prediction: 移除填充后的预测结果
    """
    pad_left, pad_top, pad_right, pad_bottom = padding_info
    original_width, original_height = original_size

    # 裁剪掉填充部分
    unpadded = prediction[pad_top:pad_top+original_height, pad_left:pad_left+original_width]

    return unpadded


def sliding_window_predict(model, image, window_size=512, stride=256, device='cuda',
                          skip_empty_windows=True, valid_content_threshold=0.1,
                          pad_to_multiple=True, pad_fill_color='edge_mean'):
    """
    使用滑动窗口对大图进行预测

    Args:
        model: 模型
        image: PIL图像
        window_size: 窗口大小
        stride: 滑动步长
        device: 设备
        skip_empty_windows: 是否跳过空白窗口（用于椭圆形等不规则图片）
        valid_content_threshold: 有效内容阈值（窗口中有效像素占比需大于此值才处理）
        pad_to_multiple: 是否将图像填充到window_size的倍数
        pad_fill_color: 填充颜色策略
    """
    original_image = image
    original_size = image.size
    padding_info = (0, 0, 0, 0)
    padding_mask = None

    # 如果启用填充，先将图像填充到window_size的倍数
    if pad_to_multiple:
        image, padding_info = pad_image_to_multiple(image, multiple=window_size, fill_color=pad_fill_color)

        # 创建填充掩码（用于后续移除填充区域的检测结果）
        if padding_info != (0, 0, 0, 0):
            padding_mask = create_padding_mask(image.size, padding_info, original_size)
            print(f"  创建填充掩码，将在预测后移除填充区域的检测结果")

    width, height = image.size
    prediction = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 如果启用跳过空白窗口，先检测有效区域
    valid_mask = None
    if skip_empty_windows:
        valid_mask, _ = detect_valid_region(image, threshold=10)

    n_windows_h = (height - window_size) // stride + 1
    n_windows_w = (width - window_size) // stride + 1

    if height < window_size:
        n_windows_h = 1
    if width < window_size:
        n_windows_w = 1

    total_windows = n_windows_h * n_windows_w
    print(f"图片尺寸: {width}x{height}")
    print(f"窗口大小: {window_size}x{window_size}")
    print(f"滑动步长: {stride}")
    print(f"窗口数量: {n_windows_w}x{n_windows_h} = {total_windows}")
    if skip_empty_windows:
        print(f"跳过空白窗口: 启用 (有效内容阈值: {valid_content_threshold*100:.1f}%)")

    window_count = 0
    skipped_count = 0

    for i in range(n_windows_h):
        for j in range(n_windows_w):
            if height >= window_size:
                y_start = min(i * stride, height - window_size)
            else:
                y_start = 0

            if width >= window_size:
                x_start = min(j * stride, width - window_size)
            else:
                x_start = 0

            y_end = min(y_start + window_size, height)
            x_end = min(x_start + window_size, width)

            actual_h = y_end - y_start
            actual_w = x_end - x_start

            # 检查窗口是否包含足够的有效内容
            if skip_empty_windows and valid_mask is not None:
                window_valid_mask = valid_mask[y_start:y_end, x_start:x_end]
                valid_ratio = np.sum(window_valid_mask) / (actual_h * actual_w)

                if valid_ratio < valid_content_threshold:
                    # 跳过这个窗口
                    skipped_count += 1
                    continue

            window = image.crop((x_start, y_start, x_end, y_end))

            if window.size != (window_size, window_size):
                padded = Image.new('RGB', (window_size, window_size), (0, 0, 0))
                padded.paste(window, (0, 0))
                window = padded

            window_tensor = transform(window).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(window_tensor)

            pred = output[0, 0, ...].cpu().numpy()
            pred = pred[:actual_h, :actual_w]
            
            prediction[y_start:y_end, x_start:x_end] += pred
            count_map[y_start:y_end, x_start:x_end] += 1

            window_count += 1
            if window_count % 10 == 0:
                processed = window_count + skipped_count
                print(f"  处理进度: {processed}/{total_windows} ({processed/total_windows*100:.1f}%) - 已处理: {window_count}, 已跳过: {skipped_count}")

    # 最终统计
    processed_total = window_count + skipped_count
    print(f"  完成! 总窗口: {total_windows}, 已处理: {window_count}, 已跳过: {skipped_count}")

    prediction = np.divide(prediction, count_map, where=count_map > 0)

    # 如果进行了填充，应用掩码移除填充区域的检测结果
    if pad_to_multiple and padding_info != (0, 0, 0, 0):
        if padding_mask is not None:
            print(f"  应用填充掩码，移除填充区域的检测结果")
            # 将填充区域的预测值设为0
            prediction = prediction * padding_mask

        print(f"  移除填充，恢复原始尺寸: {original_size[0]}x{original_size[1]}")
        prediction = unpad_prediction(prediction, padding_info, original_size)
        count_map = unpad_prediction(count_map, padding_info, original_size)

    return prediction, count_map

def remove_ruler_from_mask(crack_mask, ruler_mask):
    """从裂缝mask中移除尺子区域"""
    # 扩展尺子mask，确保完全移除
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    ruler_mask_expanded = cv2.dilate(ruler_mask, kernel, iterations=2)
    
    # 从裂缝mask中减去尺子区域
    crack_mask_clean = cv2.bitwise_and(crack_mask, cv2.bitwise_not(ruler_mask_expanded))
    
    return crack_mask_clean

def extract_crack_skeletons(binary_mask):
    """
    提取裂缝的骨架
    
    Args:
        binary_mask: 二值化的裂缝mask
    
    Returns:
        skeleton: 骨架图
    """
    # 确保是二值图
    binary = (binary_mask > 0).astype(np.uint8)
    
    # 骨架化
    skeleton = morphology.skeletonize(binary)
    
    return skeleton.astype(np.uint8)

def measure_crack_lengths(skeleton, pixel_to_cm, min_length_pixels=20):
    """
    测量每条裂缝的长度
    
    Args:
        skeleton: 骨架图
        pixel_to_cm: 像素到厘米的转换比例
        min_length_pixels: 最小裂缝长度（像素），小于此值的忽略
    
    Returns:
        crack_info: 裂缝信息列表
    """
    # 标记连通区域
    labeled_skeleton = measure.label(skeleton, connectivity=2)
    
    crack_info = []
    
    # 遍历每个连通区域
    for region in measure.regionprops(labeled_skeleton):
        # 获取区域的像素数量（骨架长度的近似）
        length_pixels = region.area
        
        # 过滤太小的区域
        if length_pixels < min_length_pixels:
            continue
        
        # 转换为厘米
        length_cm = length_pixels * pixel_to_cm
        
        # 获取边界框
        minr, minc, maxr, maxc = region.bbox
        
        crack_info.append({
            'id': region.label,
            'length_pixels': length_pixels,
            'length_cm': length_cm,
            'bbox': (minc, minr, maxc - minc, maxr - minr),  # (x, y, w, h)
            'centroid': region.centroid
        })
    
    # 按长度排序
    crack_info.sort(key=lambda x: x['length_cm'], reverse=True)
    
    return crack_info

def create_visualization(original_img, crack_mask, ruler_bbox, crack_info, pixel_to_cm, ruler_length_cm=50,
                        test_lines=None, rqd_results=None):
    """创建可视化结果"""
    img_np = np.array(original_img)
    vis = img_np.copy()

    # 绘制尺子边界框（绿色）
    if ruler_bbox is not None:
        x, y, w, h = ruler_bbox
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 4)
        cv2.putText(vis, f"Ruler: {ruler_length_cm}cm", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # 绘制裂缝（半透明红色）
    crack_colored = np.zeros_like(img_np)
    crack_colored[:, :, 2] = crack_mask  # 红色通道
    vis = cv2.addWeighted(vis, 0.7, crack_colored, 0.3, 0)

    # 绘制每条裂缝的标注
    for i, crack in enumerate(crack_info):
        x, y, w, h = crack['bbox']
        cx, cy = int(crack['centroid'][1]), int(crack['centroid'][0])

        # 根据裂缝长度选择颜色
        if crack['length_cm'] > 30:
            color = (0, 0, 255)  # 红色 - 长裂缝
            thickness = 3
        elif crack['length_cm'] > 15:
            color = (0, 165, 255)  # 橙色 - 中等裂缝
            thickness = 2
        else:
            color = (0, 255, 255)  # 黄色 - 短裂缝
            thickness = 2

        # 绘制边界框
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, thickness)

        # 绘制中心点
        cv2.circle(vis, (cx, cy), 5, color, -1)

        # 绘制标签（带背景）
        label = f"#{i+1}: {crack['length_cm']:.1f}cm"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2

        # 计算文本大小
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # 确定标签位置（避免超出图像边界）
        label_x = max(x, 5)
        label_y = max(y - 10, text_h + 10)

        # 绘制文本背景（黑色半透明）
        overlay = vis.copy()
        cv2.rectangle(overlay,
                     (label_x - 5, label_y - text_h - 5),
                     (label_x + text_w + 5, label_y + baseline + 5),
                     (0, 0, 0), -1)
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        # 绘制文本
        cv2.putText(vis, label, (label_x, label_y),
                   font, font_scale, color, font_thickness)

    # 添加图例
    legend_y = 30
    legend_x = vis.shape[1] - 250

    # 图例背景
    overlay = vis.copy()
    cv2.rectangle(overlay, (legend_x - 10, 10), (vis.shape[1] - 10, 150), (0, 0, 0), -1)
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    # 图例文本
    cv2.putText(vis, "Crack Length:", (legend_x, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "> 30cm", (legend_x, legend_y + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(vis, "15-30cm", (legend_x, legend_y + 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    cv2.putText(vis, "< 15cm", (legend_x, legend_y + 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 添加统计信息
    stats_y = vis.shape[0] - 100
    stats_x = 20

    # 统计背景
    overlay = vis.copy()
    cv2.rectangle(overlay, (10, stats_y - 30), (350, vis.shape[0] - 10), (0, 0, 0), -1)
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    # 统计文本
    total_length = sum(c['length_cm'] for c in crack_info)
    cv2.putText(vis, f"Total Cracks: {len(crack_info)}", (stats_x, stats_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Total Length: {total_length:.1f} cm", (stats_x, stats_y + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if len(crack_info) > 0:
        cv2.putText(vis, f"Avg Length: {total_length/len(crack_info):.1f} cm", (stats_x, stats_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 绘制RQD测线（如果提供）
    if test_lines is not None and rqd_results is not None:
        # 绘制4条测线（虚线，黄色）
        line_names_en = ['H-Line1', 'H-Line2', 'V-Line1', 'V-Line2']

        for i, (start, end) in enumerate(test_lines):
            # 绘制虚线
            draw_dashed_line(vis, start, end, (255, 255, 0), thickness=3, dash_length=15)

            # 在测线中点标注测线名称
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2

            # 根据横线/竖线调整标签位置
            if i < 2:  # 横线
                label_x = mid_x - 50
                label_y = mid_y - 15
            else:  # 竖线
                label_x = mid_x + 15
                label_y = mid_y

            label = line_names_en[i]

            # 绘制标签背景
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            overlay = vis.copy()
            cv2.rectangle(overlay,
                         (label_x - 5, label_y - text_h - 5),
                         (label_x + text_w + 5, label_y + 5),
                         (0, 0, 0), -1)
            vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

            # 绘制标签文本
            cv2.putText(vis, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 添加节理间距统计信息（右下角）
        spacing_y = vis.shape[0] - 160
        spacing_x = vis.shape[1] - 350

        # 背景
        overlay = vis.copy()
        cv2.rectangle(overlay, (spacing_x - 10, spacing_y - 30), (vis.shape[1] - 10, vis.shape[0] - 10), (0, 0, 0), -1)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        # 标题
        cv2.putText(vis, "Joint Spacing:", (spacing_x, spacing_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 每条测线的节理间距（使用英文标签）
        for i, result in enumerate(rqd_results):
            text = f"{line_names_en[i]}: {result['joint_spacing_cm']:.1f}cm"
            cv2.putText(vis, text, (spacing_x, spacing_y + 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 平均节理间距
        joint_spacing_avg = np.mean([r['joint_spacing_cm'] for r in rqd_results])
        cv2.putText(vis, f"Avg: {joint_spacing_avg:.1f}cm", (spacing_x, spacing_y + 30 + len(rqd_results)*25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    return vis

def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    """绘制虚线"""
    x1, y1 = pt1
    x2, y2 = pt2

    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    dashes = int(dist / dash_length)

    if dashes == 0:
        cv2.line(img, pt1, pt2, color, thickness)
        return

    for i in range(dashes):
        if i % 2 == 0:  # 只绘制偶数段
            start_ratio = i / dashes
            end_ratio = min((i + 1) / dashes, 1.0)

            start_x = int(x1 + (x2 - x1) * start_ratio)
            start_y = int(y1 + (y2 - y1) * start_ratio)
            end_x = int(x1 + (x2 - x1) * end_ratio)
            end_y = int(y1 + (y2 - y1) * end_ratio)

            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Crack Length Measurement with Ruler Detection')
    parser.add_argument('--image_path', type=str, default=None, help='输入图片路径（单张图片）')
    parser.add_argument('--input_dir', type=str, default='data/cut_picture',
                       help='输入图片目录（批量处理）')
    parser.add_argument('--model_path', type=str,
                       default='./checkpoints/weights/checkpoint_best.pth',
                       help='模型权重文件路径')
    parser.add_argument('--output_base_dir', type=str, default='data/two-step-result',
                       help='输出基础目录')
    parser.add_argument('--enhanced_save_dir', type=str, default='data/one-step-images',
                       help='增强图片保存目录')
    parser.add_argument('--ruler_length', type=float, default=50.0,
                       help='尺子的实际长度(cm)')
    parser.add_argument('--device', type=str, default='cuda', help='设备: cuda 或 cpu')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值 (0-1)')
    parser.add_argument('--window_size', type=int, default=512, help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=256, help='滑动步长')
    parser.add_argument('--min_crack_length', type=int, default=20,
                       help='最小裂缝长度(像素)，小于此值的忽略')
    parser.add_argument('--interactive_roi', action='store_true',
                       help='启用交互式ROI选择（识别完成后可以圈选感兴趣区域）')
    parser.add_argument('--roi_coords', type=str, default=None,
                       help='手动指定ROI坐标，格式: "x1,y1;x2,y2;x3,y3;..." 例如: "100,100;500,100;500,400;100,400"')
    parser.add_argument('--calculate_rqd', action='store_true',
                       help='计算RQD值（岩石质量指标）')
    parser.add_argument('--rqd_threshold', type=float, default=20.0,
                       help='RQD计算阈值(cm)，默认20cm')
    parser.add_argument('--detect_valid_region', action='store_true',
                       help='检测有效区域（用于椭圆形等不规则裁剪图片）')
    parser.add_argument('--valid_region_threshold', type=int, default=10,
                       help='有效区域检测阈值（像素值），默认10')
    parser.add_argument('--valid_content_threshold', type=float, default=0.1,
                       help='滑动窗口有效内容阈值（0-1），窗口中有效像素占比需大于此值才处理，默认0.1')
    parser.add_argument('--edge_erode_size', type=int, default=15,
                       help='边缘腐蚀核大小（像素），用于移除边缘区域的误检测，默认15')
    parser.add_argument('--pad_to_multiple', action='store_true',
                       help='将图像填充到窗口大小的倍数（推荐用于不规则图片）')
    parser.add_argument('--pad_fill_color', type=str, default='edge_mean',
                       choices=['edge_mean', 'white', 'gray'],
                       help='填充颜色策略: edge_mean=边缘平均色(默认), white=白色, gray=灰色')
    parser.add_argument('--enhance', action='store_true',
                       help='启用图像增强（Retinex增强），增强后的图片会保存到输出目录')
    parser.add_argument('--BCELoss_ratio', default=0.2, type=float)
    parser.add_argument('--DiceLoss_ratio', default=0.8, type=float)
    parser.add_argument('--Norm_Type', default='GN', type=str)

    args = parser.parse_args()

    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        exit(1)

    # 确定输入图片列表
    image_paths = []

    if args.image_path:
        # 单张图片模式
        if not os.path.exists(args.image_path):
            print(f"错误: 图片文件不存在: {args.image_path}")
            exit(1)
        image_paths = [args.image_path]
    else:
        # 批量处理模式
        if not os.path.exists(args.input_dir):
            print(f"错误: 输入目录不存在: {args.input_dir}")
            exit(1)

        # 获取目录中所有图片文件
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        for file in os.listdir(args.input_dir):
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_formats:
                image_paths.append(os.path.join(args.input_dir, file))

        if len(image_paths) == 0:
            print(f"错误: 在目录 {args.input_dir} 中未找到图片文件")
            exit(1)

        print(f"找到 {len(image_paths)} 张图片待处理")

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ 警告: CUDA不可用，将使用CPU")

    # 加载模型（只加载一次）
    print("\n正在加载模型...")
    try:
        model, _ = build_model(args)
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict["model"])
        model.to(device)
        model.eval()
        print("✓ 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # 统计信息
    success_count = 0
    failed_count = 0
    skipped_count = 0

    # 处理每张图片
    for idx, image_path in enumerate(image_paths):
        print("\n" + "=" * 80)
        print(f"处理图片 [{idx+1}/{len(image_paths)}]: {os.path.basename(image_path)}")
        print("=" * 80)

        # 创建输出目录
        try:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_image_name = image_name.replace('.', '_').replace('/', '_').replace('\\', '_')
            output_dir = os.path.join(args.output_base_dir, f"{safe_image_name}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"❌ 创建输出目录失败: {e}")
            failed_count += 1
            continue

        print(f"输出目录: {output_dir}")
        print(f"尺子长度: {args.ruler_length} cm")

        try:
            # 读取图片
            print("\n[1/7] 正在读取图片..." if args.enhance else "\n[1/6] 正在读取图片...")

            # 检查文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {image_path}")

            # 检查文件大小
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                raise ValueError(f"图片文件为空: {image_path}")

            original_img_cv = cv2.imread(image_path)
            if original_img_cv is None:
                raise ValueError(f"无法读取图片，可能是格式不支持或文件损坏")

            original_img = Image.open(image_path).convert('RGB')
            if original_img.size[0] == 0 or original_img.size[1] == 0:
                raise ValueError(f"图片尺寸无效: {original_img.size}")

            print(f"✓ 图片尺寸: {original_img.size[0]}x{original_img.size[1]}")
            print(f"✓ 文件大小: {file_size / 1024:.1f} KB")

            # 图像增强（如果启用）
            if args.enhance:
                print("\n[2/6] 正在进行图像增强...")
                try:
                    enhanced_img_cv = cracknex_retinex_enhancement(original_img_cv)

                    # 保存一份到输出目录
                    enhanced_path_output = os.path.join(output_dir, f"{image_name}_enhanced.jpg")
                    cv2.imwrite(enhanced_path_output, enhanced_img_cv)
                    print(f"✓ 增强图片已保存")

                    # 使用增强后的图片进行后续处理
                    original_img = Image.fromarray(cv2.cvtColor(enhanced_img_cv, cv2.COLOR_BGR2RGB))
                    print(f"✓ 将使用增强后的图片进行检测")
                except Exception as e:
                    print(f"⚠ 图像增强失败: {e}，将使用原始图片")
                    # 继续使用原始图片

            # 检测尺子
            step_num = "3/7" if args.enhance else "2/6"
            print(f"\n[{step_num}] 正在检测尺子...")
            try:
                pixel_to_cm, ruler_mask, ruler_bbox = detect_ruler(original_img, args.ruler_length)
            except Exception as e:
                print(f"❌ 尺子检测异常: {e}")
                pixel_to_cm = None
                ruler_mask = None
                ruler_bbox = None

            if pixel_to_cm is None:
                print(f"⚠ 警告: 无法检测到尺子，跳过此图片!")
                skipped_count += 1
                continue

            # 滑动窗口预测
            step_num = "4/7" if args.enhance else "3/6"
            print(f"\n[{step_num}] 正在进行裂缝检测...")
            try:
                start_time = time.time()
                prediction, count_map = sliding_window_predict(
                    model, original_img,
                    window_size=args.window_size,
                    stride=args.stride,
                    device=device,
                    skip_empty_windows=args.detect_valid_region,  # 如果启用有效区域检测，则跳过空白窗口
                    valid_content_threshold=args.valid_content_threshold,  # 窗口有效内容阈值
                    pad_to_multiple=args.pad_to_multiple,  # 是否填充到窗口大小的倍数
                    pad_fill_color=args.pad_fill_color  # 填充颜色策略
                )
                inference_time = time.time() - start_time
                print(f"✓ 预测完成! 耗时: {inference_time:.2f}s")
            except Exception as e:
                print(f"❌ 裂缝检测失败: {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue

            # 后处理
            try:
                pred_norm = prediction / np.max(prediction) if np.max(prediction) > 0 else prediction
                pred_binary = ((pred_norm > args.threshold) * 255).astype(np.uint8)
            except Exception as e:
                print(f"❌ 预测结果后处理失败: {e}")
                failed_count += 1
                continue

            # 如果启用了有效区域检测，先检测有效区域并移除边缘区域的检测结果
            if args.detect_valid_region:
                step_num = "4.5/7" if args.enhance else "3.5/6"
                print(f"\n[{step_num}] 正在移除边缘区域的检测结果...")
                try:
                    valid_mask_temp, _ = detect_valid_region(original_img, threshold=args.valid_region_threshold)

                    # 对有效区域掩码进行腐蚀操作，去除边缘区域
                    erode_kernel_size = args.edge_erode_size
                    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
                    valid_mask_eroded = cv2.erode(valid_mask_temp, erode_kernel, iterations=1)

                    # 将有效区域外的检测结果清零
                    pred_binary = pred_binary * valid_mask_eroded
                    print(f"  已移除边缘区域（腐蚀核大小: {erode_kernel_size}×{erode_kernel_size}）")
                except Exception as e:
                    print(f"⚠ 边缘区域移除失败: {e}")

            # 移除尺子区域
            step_num = "5/7" if args.enhance else "4/6"
            print(f"\n[{step_num}] 正在移除尺子区域...")
            crack_mask_clean = remove_ruler_from_mask(pred_binary, ruler_mask)

            # 提取骨架
            print("正在提取裂缝骨架...")
            skeleton = extract_crack_skeletons(crack_mask_clean)

            # 测量裂缝长度
            print("正在测量裂缝长度...")
            crack_info = measure_crack_lengths(skeleton, pixel_to_cm, args.min_crack_length)

            print(f"✓ 检测到 {len(crack_info)} 条裂缝")

            # 岩石质量指标计算（如果启用）
            indicator1 = None
            joint_spacing_avg = None
            indicator3 = None
            rqd_results = None
            test_lines = None

            if args.calculate_rqd:
                step_num = "5.5/7" if args.enhance else "4.5/6"
                print(f"\n[{step_num}] 正在计算岩石质量指标...")
                image_shape = (original_img.size[1], original_img.size[0])  # (height, width)

                # 检测有效区域（支持不规则形状图片）
                valid_mask = None
                valid_ratio = 1.0

                if args.detect_valid_region:
                    print("  检测图像有效区域...")
                    try:
                        valid_mask, valid_area_pixels = detect_valid_region(original_img, threshold=args.valid_region_threshold)
                        # 计算有效区域的实际面积
                        total_pixels = original_img.size[0] * original_img.size[1]
                        valid_ratio = valid_area_pixels / total_pixels if total_pixels > 0 else 1.0
                        print(f"  有效区域比例: {valid_ratio*100:.1f}%")
                    except Exception as e:
                        print(f"⚠ 有效区域检测失败: {e}")
                        valid_ratio = 1.0

                # 图像实际尺寸（cm）- 使用有效区域面积
                image_width_cm = original_img.size[0] * pixel_to_cm
                image_height_cm = original_img.size[1] * pixel_to_cm
                image_area_cm2_total = image_width_cm * image_height_cm
                image_area_cm2 = image_area_cm2_total * valid_ratio  # 有效区域面积

                # 指标1 = (长度≥25cm的裂隙条数) / 图像有效面积（m²）
                long_cracks_count = sum(1 for c in crack_info if c['length_cm'] >= 25)
                image_area_m2 = image_area_cm2 / 10000  # cm² 转 m²
                indicator1 = long_cracks_count / image_area_m2 if image_area_m2 > 0 else 0

                # 指标2 - 节理间距：计算测线与裂隙交点的平均间距
                joint_spacing_avg, rqd_results, test_lines = calculate_joint_spacing_four_lines(
                    image_shape, skeleton, pixel_to_cm
                )

                # 指标3 = 裂隙总长度（m） / 图像实际面积（m²）
                total_crack_length_cm = sum(c['length_cm'] for c in crack_info)
                total_crack_length_m = total_crack_length_cm / 100  # cm 转 m
                indicator3 = total_crack_length_m / image_area_m2 if image_area_m2 > 0 else 0

                print(f"✓ 岩石质量指标计算完成!")
                print(f"  测线布局: 2条横线 + 2条竖线 (井字形)")
                print(f"\n  图像实际尺寸: {image_width_cm:.1f} cm × {image_height_cm:.1f} cm")

                if args.detect_valid_region:
                    print(f"  图像总面积: {image_area_cm2_total:.1f} cm²")
                    print(f"  有效区域比例: {valid_ratio*100:.1f}%")
                    print(f"  有效区域面积: {image_area_cm2:.1f} cm² ({image_area_m2:.4f} m²)")
                else:
                    print(f"  图像面积: {image_area_cm2:.1f} cm² ({image_area_m2:.4f} m²)")

                print(f"  裂隙总数: {len(crack_info)} 条")
                print(f"  长度≥25cm的裂隙: {long_cracks_count} 条")
                print(f"  裂隙总长度: {total_crack_length_cm:.1f} cm ({total_crack_length_m:.2f} m)")
                print(f"\n  指标1: {indicator1:.4f} 条/m²")
                print(f"  指标2 - 平均节理间距: {joint_spacing_avg:.2f} cm")
                print(f"  指标3: {indicator3:.4f} m/m²")

            # 区域选择（交互式或手动指定）
            roi_mask = None
            roi_points = []
            crack_info_filtered = crack_info

            # 方式1: 手动指定ROI坐标
            if args.roi_coords and len(crack_info) > 0:
                step_num = "6/7" if args.enhance else "5/6"
                print(f"\n[{step_num}] 使用手动指定的ROI坐标...")
                try:
                    # 解析坐标字符串 "x1,y1;x2,y2;x3,y3;..."
                    coords_str = args.roi_coords.split(';')
                    roi_points = []
                    for coord in coords_str:
                        x, y = map(int, coord.strip().split(','))
                        roi_points.append([x, y])

                    if len(roi_points) >= 3:
                        # 创建ROI mask
                        roi_mask = np.zeros((original_img.size[1], original_img.size[0]), dtype=np.uint8)
                        pts = np.array(roi_points, dtype=np.int32)
                        cv2.fillPoly(roi_mask, [pts], 255)

                        # 根据ROI过滤裂缝
                        crack_info_filtered = filter_cracks_by_roi(crack_info, skeleton, roi_mask)
                        print(f"✓ ROI顶点数: {len(roi_points)}")
                        print(f"✓ ROI内裂缝数量: {len(crack_info_filtered)} / {len(crack_info)}")
                    else:
                        print("⚠ ROI坐标点少于3个，忽略ROI设置")
                        roi_points = []
                except Exception as e:
                    print(f"⚠ 解析ROI坐标失败: {e}")
                    print("  格式示例: --roi_coords \"100,100;500,100;500,400;100,400\"")
                    roi_points = []

            # 方式2: 交互式选择ROI（需要图形界面）
            elif args.interactive_roi and len(crack_info) > 0:
                step_num = "6/7" if args.enhance else "5/6"
                print(f"\n[{step_num}] 交互式区域选择...")

                try:
                    # 创建预览图像用于选择
                    preview_img = np.array(original_img).copy()

                    # 在预览图上显示所有检测到的裂缝（半透明）
                    crack_overlay = np.zeros_like(preview_img)
                    crack_overlay[crack_mask_clean > 0] = [0, 255, 0]  # 绿色
                    preview_img = cv2.addWeighted(preview_img, 0.7, crack_overlay, 0.3, 0)

                    # 显示尺子位置
                    if ruler_bbox is not None:
                        x, y, w, h = ruler_bbox
                        cv2.rectangle(preview_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    # 让用户选择ROI
                    roi_mask, roi_points = select_roi_interactive(preview_img)

                    if roi_mask is not None:
                        # 根据ROI过滤裂缝
                        crack_info_filtered = filter_cracks_by_roi(crack_info, skeleton, roi_mask)
                        print(f"✓ ROI内裂缝数量: {len(crack_info_filtered)} / {len(crack_info)}")
                    else:
                        print("✓ 未选择ROI，使用全部裂缝")
                except Exception as e:
                    print(f"⚠ 交互式ROI选择失败: {e}")
                    print("  提示: 如果没有图形界面，请使用 --roi_coords 参数手动指定坐标")
                    print("  示例: --roi_coords \"100,100;500,100;500,400;100,400\"")
                    roi_points = []

            # 保存结果
            step_num = "7/7" if args.enhance else "6/6"
            print(f"\n[{step_num}] 正在保存结果...")

            try:
                # 1. 保存原图
                original_output_path = os.path.join(output_dir, f"{image_name}_original.jpg")
                # 如果启用了增强，保存原始图片（未增强的）
                if args.enhance:
                    original_img_pil = Image.open(image_path).convert('RGB')
                    original_img_pil.save(original_output_path)
                else:
                    original_img.save(original_output_path)
                print(f"✓ 已保存: {image_name}_original.jpg")
            except Exception as e:
                print(f"⚠ 保存原图失败: {e}")

            try:
                # 2. 保存尺子检测结果
                ruler_vis = np.array(original_img).copy()
                if ruler_bbox is not None:
                    x, y, w, h = ruler_bbox
                    cv2.rectangle(ruler_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(ruler_vis, f"Ruler: {args.ruler_length}cm", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                ruler_output_path = os.path.join(output_dir, f"{image_name}_ruler_detection.jpg")
                cv2.imwrite(ruler_output_path, cv2.cvtColor(ruler_vis, cv2.COLOR_RGB2BGR))
                print(f"✓ 已保存: {image_name}_ruler_detection.jpg")
            except Exception as e:
                print(f"⚠ 保存尺子检测结果失败: {e}")

            try:
                # 3. 保存裂缝预测结果
                crack_output_path = os.path.join(output_dir, f"{image_name}_crack_prediction.png")
                cv2.imwrite(crack_output_path, pred_binary)
                print(f"✓ 已保存: {image_name}_crack_prediction.png")
            except Exception as e:
                print(f"⚠ 保存裂缝预测结果失败: {e}")

            try:
                # 4. 保存清理后的裂缝mask
                crack_clean_output_path = os.path.join(output_dir, f"{image_name}_crack_clean.png")
                cv2.imwrite(crack_clean_output_path, crack_mask_clean)
                print(f"✓ 已保存: {image_name}_crack_clean.png")
            except Exception as e:
                print(f"⚠ 保存清理后的裂缝mask失败: {e}")

            try:
                # 5. 保存骨架
                skeleton_output_path = os.path.join(output_dir, f"{image_name}_skeleton.png")
                cv2.imwrite(skeleton_output_path, skeleton * 255)
                print(f"✓ 已保存: {image_name}_skeleton.png")
            except Exception as e:
                print(f"⚠ 保存骨架图失败: {e}")

            # 5b. 保存节理间距计算过程可视化（如果计算了指标）
            if rqd_results is not None and test_lines is not None:
                spacing_vis = visualize_joint_spacing(skeleton, rqd_results, test_lines, pixel_to_cm)
                spacing_vis_output_path = os.path.join(output_dir, f"{image_name}_joint_spacing.jpg")
                cv2.imwrite(spacing_vis_output_path, spacing_vis)
                print(f"✓ 节理间距计算过程可视化: {image_name}_joint_spacing.jpg")

                # 5c. 在可视化图上添加三大指标汇总面板
                if indicator3 is not None and indicator1 is not None:
                    # 在左下角添加三大指标汇总面板
                    panel_x = 30
                    panel_y = spacing_vis.shape[0] - 150
                    panel_w = 420
                    panel_h = 120

                    # 绘制半透明背景
                    overlay = spacing_vis.copy()
                    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.8, spacing_vis, 0.2, 0, spacing_vis)

                    # 绘制边框
                    cv2.rectangle(spacing_vis, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), (0, 255, 255), 3)

                    # 标题
                    cv2.putText(spacing_vis, "Rock Quality Indicators", (panel_x+10, panel_y+28),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                    # 三大指标
                    cv2.putText(spacing_vis, f"1. Indicator 1: {indicator1:.4f} cracks/m2", (panel_x+15, panel_y+58),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(spacing_vis, f"2. Avg Joint Spacing: {joint_spacing_avg:.2f} cm", (panel_x+15, panel_y+83),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(spacing_vis, f"3. Indicator 3: {indicator3:.4f} m/m2", (panel_x+15, panel_y+108),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)

                    # 重新保存带三大指标的可视化图
                    cv2.imwrite(spacing_vis_output_path, spacing_vis)

            # 6. 保存可视化结果（全部裂缝）
            vis_all = create_visualization(original_img, crack_mask_clean, ruler_bbox, crack_info, pixel_to_cm, args.ruler_length,
                                           test_lines=test_lines, rqd_results=rqd_results)

            vis_all_output_path = os.path.join(output_dir, f"{image_name}_visualization_all.jpg")
            cv2.imwrite(vis_all_output_path, cv2.cvtColor(vis_all, cv2.COLOR_RGB2BGR))

            # 6b. 保存ROI区域的可视化结果（如果有ROI）
            if roi_mask is not None and len(crack_info_filtered) > 0:
                vis_roi = create_visualization(original_img, crack_mask_clean, ruler_bbox,
                                               crack_info_filtered, pixel_to_cm, args.ruler_length,
                                               test_lines=test_lines, rqd_results=rqd_results)

                # 在可视化图上绘制ROI边界
                if len(roi_points) > 0:
                    pts = np.array(roi_points, dtype=np.int32)
                    cv2.polylines(vis_roi, [pts], True, (255, 0, 255), 3)  # 紫色边界
                    # 添加ROI标签
                    cv2.putText(vis_roi, "ROI", (roi_points[0][0], roi_points[0][1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)

                vis_roi_output_path = os.path.join(output_dir, f"{image_name}_visualization_roi.jpg")
                cv2.imwrite(vis_roi_output_path, cv2.cvtColor(vis_roi, cv2.COLOR_RGB2BGR))
                print(f"✓ ROI可视化结果: {image_name}_visualization_roi.jpg")

                # 主可视化使用ROI过滤后的结果
                vis = vis_roi
                vis_output_path = vis_roi_output_path
            else:
                # 没有ROI，使用全部裂缝
                vis = vis_all
                vis_output_path = vis_all_output_path

            # 6c. 保存简化版可视化（只显示前20条最长的裂缝，更清晰）
            display_cracks = crack_info_filtered if roi_mask is not None else crack_info
            if len(display_cracks) > 20:
                vis_top20 = create_visualization(original_img, crack_mask_clean, ruler_bbox,
                                                 display_cracks[:20], pixel_to_cm, args.ruler_length,
                                                 test_lines=test_lines, rqd_results=rqd_results)

                # 如果有ROI，也绘制ROI边界
                if roi_mask is not None and len(roi_points) > 0:
                    pts = np.array(roi_points, dtype=np.int32)
                    cv2.polylines(vis_top20, [pts], True, (255, 0, 255), 3)

                vis_top20_path = os.path.join(output_dir, f"{image_name}_visualization_top20.jpg")
                cv2.imwrite(vis_top20_path, cv2.cvtColor(vis_top20, cv2.COLOR_RGB2BGR))
                print(f"✓ 可视化结果(前20条): {image_name}_visualization_top20.jpg")

            # 7. 保存详细报告
            report_path = os.path.join(output_dir, f"{image_name}_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("裂缝长度测量报告\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"图片: {image_path}\n")
                f.write(f"时间: {timestamp}\n")
                f.write(f"图片尺寸: {original_img.size[0]} x {original_img.size[1]} 像素\n\n")

                f.write("尺子信息:\n")
                f.write(f"  实际长度: {args.ruler_length} cm\n")
                if ruler_bbox:
                    f.write(f"  检测位置: {ruler_bbox}\n")
                f.write(f"  像素比例: 1 像素 = {pixel_to_cm:.4f} cm\n\n")

                # 岩石质量指标信息
                if indicator1 is not None and rqd_results is not None:
                    f.write("=" * 60 + "\n")
                    f.write("岩石质量分析 (三大指标)\n")
                    f.write("=" * 60 + "\n\n")

                    # 计算图像实际尺寸和面积
                    image_width_cm_val = original_img.size[0] * pixel_to_cm
                    image_height_cm_val = original_img.size[1] * pixel_to_cm
                    image_area_cm2_val = image_width_cm_val * image_height_cm_val
                    image_area_m2_val = image_area_cm2_val / 10000

                    total_crack_length_cm = sum(c['length_cm'] for c in crack_info)
                    total_crack_length_m = total_crack_length_cm / 100
                    long_cracks_count = sum(1 for c in crack_info if c['length_cm'] >= 25)

                    f.write("图像实际尺寸:\n")
                    f.write(f"  宽度: {image_width_cm_val:.1f} cm\n")
                    f.write(f"  高度: {image_height_cm_val:.1f} cm\n")
                    f.write(f"  面积: {image_area_cm2_val:.1f} cm² ({image_area_m2_val:.4f} m²)\n")

                    f.write(f"  裂隙总数: {len(crack_info)} 条\n")
                    f.write(f"  长度≥25cm的裂隙: {long_cracks_count} 条\n")
                    f.write(f"  裂隙总长度: {total_crack_length_cm:.1f} cm ({total_crack_length_m:.2f} m)\n\n")

                    f.write("计算方法:\n")
                    f.write(f"  指标1: (长度≥25cm的裂隙条数) / 图像实际面积（m²）\n")
                    f.write(f"  指标2 - 节理间距: (第一个交点到最后一个交点的距离) / 交点个数\n")
                    f.write(f"  指标3: 裂隙总长度（m） / 图像实际面积（m²）\n")
                    f.write(f"  测线布局: 2条横线 + 2条竖线 (井字形)\n\n")

                    f.write("各测线节理间距统计:\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"{'测线':<10} {'节理间距(cm)':<15} {'有效长度(cm)':<15} {'交点数':<10}\n")
                    f.write("-" * 70 + "\n")

                    for result in rqd_results:
                        f.write(f"{result['line_name']:<10} {result['joint_spacing_cm']:<15.1f} "
                               f"{result['effective_length_cm']:<15.1f} {result['num_intersections']:<10}\n")

                    f.write("-" * 70 + "\n")
                    f.write("\n三大指标汇总:\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"  指标1: {indicator1:.4f} 条/m²\n")
                    f.write(f"  指标2 - 平均节理间距: {joint_spacing_avg:.2f} cm\n")
                    f.write(f"  指标3: {indicator3:.4f} m/m²\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("注: 有效长度 = 第一个交点到最后一个交点的距离\n\n")

                    # 详细交点信息
                    f.write("各测线详细交点信息:\n")
                    for result in rqd_results:
                        f.write(f"\n{result['line_name']} (有效长度: {result['effective_length_cm']:.1f}cm, 交点数: {result['num_intersections']}):\n")
                        if len(result['intersections']) == 0:
                            f.write(f"  无交点\n")
                        else:
                            for i, (x, y, dist) in enumerate(result['intersections']):
                                f.write(f"  交点{chr(65+i)}: 位置({x:.1f}, {y:.1f}), 距离起点: {dist:.1f}cm\n")

                    f.write("\n" + "=" * 60 + "\n\n")

                # ROI信息
                if roi_mask is not None:
                    f.write("ROI (感兴趣区域):\n")
                    f.write(f"  已选择ROI区域\n")
                    f.write(f"  ROI顶点数: {len(roi_points)}\n")
                    f.write(f"  全图裂缝数: {len(crack_info)}\n")
                    f.write(f"  ROI内裂缝数: {len(crack_info_filtered)}\n\n")

                # 使用过滤后的裂缝信息
                report_cracks = crack_info_filtered if roi_mask is not None else crack_info

                f.write(f"{'ROI内' if roi_mask is not None else ''}检测到 {len(report_cracks)} 条裂缝:\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'ID':<5} {'长度(cm)':<12} {'长度(像素)':<12} {'位置(x,y,w,h)'}\n")
                f.write("-" * 60 + "\n")

                total_length_cm = 0
                for i, crack in enumerate(report_cracks):
                    f.write(f"{i+1:<5} {crack['length_cm']:<12.2f} {crack['length_pixels']:<12} "
                           f"{crack['bbox']}\n")
                    total_length_cm += crack['length_cm']

                f.write("-" * 60 + "\n")
                f.write(f"总长度: {total_length_cm:.2f} cm\n")
                f.write(f"平均长度: {total_length_cm/len(report_cracks):.2f} cm\n" if len(report_cracks) > 0 else "")
                f.write(f"\n推理时间: {inference_time:.2f}s\n")

            # 8. 保存CSV格式（ROI过滤后的裂缝）
            try:
                csv_path = os.path.join(output_dir, f"{image_name}_cracks{'_roi' if roi_mask is not None else ''}.csv")
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write("ID,长度(cm),长度(像素),中心X,中心Y,边界框X,边界框Y,边界框宽,边界框高\n")
                    csv_cracks = crack_info_filtered if roi_mask is not None else crack_info
                    for i, crack in enumerate(csv_cracks):
                        cx, cy = crack['centroid']
                        x, y, w, h = crack['bbox']
                        f.write(f"{i+1},{crack['length_cm']:.2f},{crack['length_pixels']},"
                               f"{cy:.1f},{cx:.1f},{x},{y},{w},{h}\n")
                print(f"✓ 已保存CSV: {os.path.basename(csv_path)}")
            except Exception as e:
                print(f"⚠ 保存CSV失败: {e}")

            # 8b. 如果有ROI，也保存全部裂缝的CSV
            if roi_mask is not None:
                try:
                    csv_all_path = os.path.join(output_dir, f"{image_name}_cracks_all.csv")
                    with open(csv_all_path, 'w', encoding='utf-8') as f:
                        f.write("ID,长度(cm),长度(像素),中心X,中心Y,边界框X,边界框Y,边界框宽,边界框高\n")
                        for i, crack in enumerate(crack_info):
                            cx, cy = crack['centroid']
                            x, y, w, h = crack['bbox']
                            f.write(f"{i+1},{crack['length_cm']:.2f},{crack['length_pixels']},"
                                   f"{cy:.1f},{cx:.1f},{x},{y},{w},{h}\n")
                    print(f"✓ 已保存CSV(全部): {os.path.basename(csv_all_path)}")
                except Exception as e:
                    print(f"⚠ 保存全部裂缝CSV失败: {e}")

            # 打印统计信息
            print("\n" + "=" * 80)
            print("测量结果:")
            print("=" * 80)

            # 显示岩石质量三大指标（如果计算了）
            if indicator1 is not None and rqd_results is not None:
                print(f"\n岩石质量分析 (三大指标):")
                print(f"  测线布局: 2条横线 + 2条竖线 (井字形)")
                print(f"\n  各测线节理间距:")
                for result in rqd_results:
                    print(f"    {result['line_name']}: 节理间距 = {result['joint_spacing_cm']:.1f}cm (交点数: {result['num_intersections']})")
                print(f"\n  ┌{'─' * 58}┐")
                print(f"  │ 指标1: {indicator1:>43.4f} 条/m² │")
                print(f"  │ 指标2 - 平均节理间距: {joint_spacing_avg:>27.2f} cm │")
                print(f"  │ 指标3: {indicator3:>44.4f} m/m² │")
                print(f"  └{'─' * 58}┘")
                print()

            # 显示全图统计
            print(f"全图检测到裂缝数量: {len(crack_info)}")

            # 如果有ROI，显示ROI统计
            if roi_mask is not None:
                print(f"ROI内裂缝数量: {len(crack_info_filtered)}")
                display_cracks = crack_info_filtered
                print(f"\n以下统计仅针对ROI内的裂缝:")
            else:
                display_cracks = crack_info

            if len(display_cracks) > 0:
                total_length = sum(c['length_cm'] for c in display_cracks)
                print(f"总长度: {total_length:.2f} cm")
                print(f"平均长度: {total_length/len(display_cracks):.2f} cm")
                print(f"最长裂缝: {display_cracks[0]['length_cm']:.2f} cm")
                print(f"最短裂缝: {display_cracks[-1]['length_cm']:.2f} cm")

                print("\n前10条最长裂缝:")
                print(f"{'ID':<5} {'长度(cm)':<12} {'长度(像素)':<12}")
                print("-" * 30)
                for i, crack in enumerate(display_cracks[:10]):
                    print(f"{i+1:<5} {crack['length_cm']:<12.2f} {crack['length_pixels']:<12}")

            print("\n" + "=" * 80)
            print("输出文件:")
            print("=" * 80)
            print(f"✓ 原图: {image_name}_original.jpg")
            print(f"✓ 尺子检测: {image_name}_ruler_detection.jpg")
            print(f"✓ 裂缝预测: {image_name}_crack_prediction.png")
            print(f"✓ 清理后裂缝: {image_name}_crack_clean.png")
            print(f"✓ 骨架图: {image_name}_skeleton.png")
            if rqd_results is not None:
                print(f"✓ 节理间距可视化: {image_name}_joint_spacing.jpg")
            print(f"✓ 可视化结果(全部): {image_name}_visualization_all.jpg")
            if roi_mask is not None:
                print(f"✓ 可视化结果(ROI): {image_name}_visualization_roi.jpg")
                print(f"✓ CSV数据(ROI): {image_name}_cracks_roi.csv")
                print(f"✓ CSV数据(全部): {image_name}_cracks_all.csv")
            else:
                print(f"✓ CSV数据: {image_name}_cracks.csv")
            print(f"✓ 详细报告: {image_name}_report.txt")
            print(f"\n✓ 所有结果已保存到: {output_dir}")

            # 成功处理
            success_count += 1

        except Exception as e:
            print(f"\n❌ 处理图片时出错: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue

    # 打印最终统计信息
    print("\n" + "=" * 80)
    print("批量处理完成!")
    print("=" * 80)
    print(f"总图片数: {len(image_paths)}")
    print(f"✓ 成功处理: {success_count} 张")
    print(f"⚠ 跳过（无尺子）: {skipped_count} 张")
    print(f"❌ 失败: {failed_count} 张")
    print(f"成功率: {success_count/len(image_paths)*100:.1f}%")
    print("=" * 80)
