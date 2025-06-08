import os
from ultralytics import YOLO


# model = YOLO("./weights/yolo11n-seg.pt")

# model.export(format="onnx")onnx

all_detections = []
# 加载模型
model = YOLO('weights/yolo11n-seg.onnx')  # 使用ONNX模型

input_dir = '/Users/hongliang/Desktop/yolov12/datasets/crack-seg/test/images'
output_dir = './output'

os.makedirs(output_dir, exist_ok=True)

# 进行预测
results = model.predict(
    source=input_dir,
    task='segment',
    imgsz=640,
    # save=True,
    # project=output_dir,
    # exist_ok=True,
    # name='exp',
)

for result in results:
    if result.boxes is not None:
        boxes_array = result.boxes.xyxy.cpu().numpy()
        boxes_list = boxes_array.tolist()
        print('box: ', boxes_list)

    if result.masks is not None:
        masks_list = result.masks.xy
        print('mask: ', masks_list)
