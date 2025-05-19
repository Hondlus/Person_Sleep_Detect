# coding = utf-8
import json
import os
from pathlib import Path
import cv2
from Crypto.SelfTest.Cipher.test_CFB import file_name


def eiseg_json_to_yolo(json_path, output_dir, img_width, img_height):
    """
    将EISEG JSON标注文件转换为YOLO格式的TXT文件

    参数:
        json_path: EISEG JSON文件路径
        output_dir: 输出目录
        img_width: 图像宽度
        img_height: 图像高度
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 获取文件名（不带扩展名）
    # file_name = Path(json_path).stem

    # 准备YOLO格式内容
    yolo_lines = []

    for i in range(len(data)):
        # 遍历每个标注对象
        label = data[i]['labelIdx']
        points = data[i]['points']
        # print(points)
        # 将多边形点转换为YOLO格式
        normalized_points = []
        for x, y in points:
            # 归一化坐标
            nx = x / img_width
            ny = y / img_height
            normalized_points.extend([nx, ny])

        # 格式: class_id x1 y1 x2 y2 ... xn yn
        yolo_line = f"{label - 1} " + " ".join([f"{p:.6f}" for p in normalized_points])
        yolo_lines.append(yolo_line)

    # 写入YOLO TXT文件
    output_path = os.path.join(output_dir, f"{file_name}.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(yolo_lines))

    print(f"转换完成: {output_path}")


# 示例用法
if __name__ == "__main__":
    # 设置参数
    image_dir = "C:/Users/dxw-user/Desktop/yolov12/datasets/owndata/images/train/"  # 替换为你的图像文件夹路径
    json_file = "C:/Users/dxw-user/Desktop/labelimg_v1.8.1/eiseg_model/label/"  # 替换为你的JSON文件路径
    output_directory = "./yolo_labels"  # 输出目录

    for i in os.listdir(image_dir):
        file_name = i.split('.')[0]
        kuozhanming = i.split('.')[1]
        img = cv2.imread(image_dir + file_name + '.' + kuozhanming)
        img_width, img_height = img.shape[1], img.shape[0]
        # print(file_name, img_width, img_height)
        eiseg_json_to_yolo(json_file + file_name + '.json', output_directory, img_width, img_height)
    # 执行转换
    # eiseg_json_to_yolo(json_file, output_directory, image_width, image_height)
