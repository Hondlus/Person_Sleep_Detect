import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def parse_yolo_annotation(annotation_path, img_width, img_height):
    """解析YOLO格式的标注文件，返回多边形列表"""
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
    """从多边形列表创建二值掩码"""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [polygon], 1)
    return mask


def combine_masks(masks, img_shape):
    """合并多个掩码为一个"""
    combined_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for mask in masks:
        # 调整掩码大小以匹配原始图像
        resized_mask = cv2.resize(mask.astype(np.float32), (img_shape[1], img_shape[0]))
        binary_mask = (resized_mask > 0.5).astype(np.uint8)
        combined_mask = np.logical_or(combined_mask, binary_mask)
    return combined_mask.astype(np.uint8)


def calculate_mask_iou(mask1, mask2):
    """计算两个掩码之间的交并比(IoU)"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def evaluate_yolo_segmentation(model_path, images_dir, labels_dir, iou_threshold=0.5):
    """
    评估YOLO分割模型的性能
    :param model_path: 模型路径
    :param images_dir: 测试图片目录
    :param labels_dir: 标注文件目录
    :param iou_threshold: IoU阈值
    :return: (高IoU结果列表, 所有结果列表)
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
            all_results.append({
                "image_path": str(img_path),
                "iou": 0,
                "above_threshold": False,
                "pred_mask": None,
                "gt_mask": gt_mask
            })
            continue

        # 获取预测掩码并合并
        pred_masks = results[0].masks.data.cpu().numpy()
        combined_pred_mask = combine_masks(pred_masks, img.shape)

        # 计算IoU
        iou = calculate_mask_iou(gt_mask, combined_pred_mask)

        # 记录结果
        result = {
            "image_path": str(img_path),
            "iou": iou,
            "above_threshold": iou > iou_threshold,
            "pred_mask": combined_pred_mask,
            "gt_mask": gt_mask
        }

        all_results.append(result)

        if iou > iou_threshold:
            high_iou_results.append(result)
            print(f"图片 {img_path.name} IoU: {iou:.4f} (大于阈值 {iou_threshold})")
        else:
            print(f"图片 {img_path.name} IoU: {iou:.4f}")
        visualize_results(result)

    return high_iou_results, all_results


def visualize_results(result):
    """可视化结果"""
    img = cv2.imread(result["image_path"])

    # 创建可视化
    overlay = img.copy()
    gt_contours, _ = cv2.findContours(result["gt_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if result["pred_mask"] is not None:
        pred_contours, _ = cv2.findContours(result["pred_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 绘制预测结果(红色)
        cv2.drawContours(overlay, pred_contours, -1, (0, 0, 255), 1)

    # 绘制真实标注(绿色)
    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 1)

    # 混合显示
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # 显示结果
    title = f"IoU={result['iou']:.2f}" + (" (Above Threshold)" if result["above_threshold"] else "")
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 配置路径
    model_path = "C:/Users/dxw-user/Desktop/yolov12/runs/segment/nseg11_混合数据集/weights/best.pt"
    images_dir = "C:/Users/dxw-user/Desktop/yolov12/datasets/mixdata/images/test/"
    labels_dir = "C:/Users/dxw-user/Desktop/yolov12/datasets/mixdata/labels/test/"

    # 设置IoU阈值
    iou_threshold = 0.5

    # 运行评估
    high_iou_results, all_results = evaluate_yolo_segmentation(
        model_path, images_dir, labels_dir, iou_threshold
    )

    # 打印结果摘要
    print("\n结果摘要:")
    print(f"共处理 {len(all_results)} 张图片")
    print(f"其中 {len(high_iou_results)} 张图片的IoU大于 {iou_threshold}")
    print(f"高IoU图片比例: {len(high_iou_results) / len(all_results):.2%}")

    # # 可视化高IoU结果
    # for sample_result in high_iou_results:
    #     visualize_results(sample_result)
