import os
from ultralytics import YOLO


# model = YOLO("./weights/best.pt")
#
# model.export(format="onnx")

# 加载模型
model = YOLO('weights/best.onnx', task='segment')  # 使用ONNX模型

input_dir = 'C:/Users/dxw-user/Desktop/yolov12/input/1616.rf.c868709931a671796794fdbb95352c5a.jpg'
output_dir = 'C:/Users/dxw-user/Desktop/yolov12/output'

os.makedirs(output_dir, exist_ok=True)

# 进行预测
results = model.predict(
    source=input_dir,
    task='segment',
    imgsz=640,
    save=True,
    project=output_dir,
    exist_ok=True,
    name='exp',
)

for result in results:
    if result.boxes is not None:
        boxes_array = result.boxes.xyxy.cpu().numpy()
        boxes_list = boxes_array.astype(int).tolist()
        print('box: ', boxes_list)

    if result.masks is not None:
        # masks_list = result.masks.xy
        masks_list = [mask.astype(int).tolist() for mask in result.masks.xy]
        print('mask: ', masks_list)
