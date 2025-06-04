import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import jaccard_score


def parse_yolo_annotation(annotation_path, img_width, img_height):
    """
    解析YOLO格式的标注文件
    返回: 包含归一化多边形坐标的列表
    """
    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    polygons = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        points = list(map(float, parts[1:]))

        # 将归一化坐标转换为绝对坐标
        absolute_points = []
        for i in range(0, len(points), 2):
            x = points[i] * img_width
            y = points[i + 1] * img_height
            absolute_points.append([x, y])

        polygons.append(np.array(absolute_points, dtype=np.int32))

    return polygons


def create_mask_from_polygons(polygons, img_shape):
    """
    从多边形创建二进制掩码
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [polygon], 1)
    return mask


def calculate_mask_iou(mask1, mask2):
    """
    计算两个掩码之间的交并比(IoU)
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def evaluate_yolo_segmentation(model_path, images_dir, labels_dir, iou_threshold=0.75):
    """
    评估YOLOv8分割模型在指定数据集上的表现
    返回: 包含高IoU结果的列表和所有结果
    """
    # 加载模型
    model = YOLO(model_path)

    # 获取所有图片路径
    image_paths = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))

    high_iou_results = []
    all_results = []

    for img_path in image_paths:
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        img_height, img_width = img.shape[:2]

        # 解析对应的标注文件
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"未找到标注文件: {label_path}")
            continue

        # 获取真实掩码
        gt_polygons = parse_yolo_annotation(label_path, img_width, img_height)
        gt_mask = create_mask_from_polygons(gt_polygons, img.shape)

        # 使用模型进行预测
        results = model(img)

        # 检查是否有分割结果
        if len(results) == 0 or results[0].masks is None:
            print(f"图片 {img_path.name} 未检测到分割结果")
            all_results.append((str(img_path), 0, False))
            continue

        # 获取预测掩码
        pred_masks = results[0].masks.data.cpu().numpy()

        # 计算每个预测掩码与真实掩码的IoU
        max_iou = 0
        best_mask = None
        for mask in pred_masks:
            # 调整掩码大小以匹配原始图像
            resized_mask = cv2.resize(mask.astype(np.float32), (img_width, img_height))
            binary_mask = (resized_mask > 0.5).astype(np.uint8)

            # 计算IoU
            iou = calculate_mask_iou(gt_mask, binary_mask)
            if iou > max_iou:
                max_iou = iou
                best_mask = binary_mask

        # 记录结果
        result = {
            "image_path": str(img_path),
            "iou": max_iou,
            "above_threshold": max_iou > iou_threshold,
            "pred_mask": best_mask,
            "gt_mask": gt_mask
        }

        all_results.append(result)

        if max_iou > iou_threshold:
            high_iou_results.append(result)
            print(f"图片 {img_path.name} IoU: {max_iou:.4f} (大于阈值 {iou_threshold})")
        else:
            print(f"图片 {img_path.name} IoU: {max_iou:.4f}")

    return high_iou_results, all_results


# 使用示例
if __name__ == "__main__":
    # 配置路径
    model_path = "C:/Users/dxw-user/Desktop/yolov12/runs/segment/nseg11_混合数据集/weights/best.pt"  # YOLOv8分割模型
    images_dir = "C:/Users/dxw-user/Desktop/yolov12/datasets/mixdata/images/test/"  # 图片目录
    labels_dir = "C:/Users/dxw-user/Desktop/yolov12/datasets/mixdata/labels/test/"  # 标注文件目录

    # 设置IoU阈值
    iou_threshold = 0.75

    # 运行评估
    high_iou_results, all_results = evaluate_yolo_segmentation(
        model_path, images_dir, labels_dir, iou_threshold
    )

    # 打印结果摘要
    print("\n结果摘要:")
    print(f"共处理 {len(all_results)} 张图片")
    print(f"其中 {len(high_iou_results)} 张图片的IoU大于 {iou_threshold}")
    print(f"高IoU图片比例: {len(high_iou_results) / len(all_results):.2%}")

    # 可选: 可视化结果
    if high_iou_results:
        sample_result = high_iou_results[0]
        img = cv2.imread(sample_result["image_path"])

        # 创建可视化
        overlay = img.copy()
        gt_contours, _ = cv2.findContours(sample_result["gt_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pred_contours, _ = cv2.findContours(sample_result["pred_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制真实标注(绿色)
        cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)
        # 绘制预测结果(红色)
        cv2.drawContours(overlay, pred_contours, -1, (0, 0, 255), 2)

        # 混合显示
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # 显示结果
        cv2.imshow(f"高IoU示例 (IoU={sample_result['iou']:.2f})", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
