from ultralytics import YOLO
import cv2
# # Load a model
# model = YOLO("yolo11n-seg.pt")  # load an official model
#
# # Export the model
# model.export(format="onnx")

onnx_model = YOLO("yolo11n-seg.onnx")

# Run inference
results = onnx_model.predict(source='./assets/bus.jpg', save=False, show=False)

if results[0].masks is not None:
    masks = results[0].masks.xy

for i in results:
    res = i.plot()  # 提取推理后的图像
    cv2.imshow("results", res)  # 显示结果
cv2.waitKey(0)
cv2.destroyAllWindows()
