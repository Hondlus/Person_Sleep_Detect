from ultralytics import YOLO
import cv2


# 裁剪图片预留的空间填充
offset = 20
# 加载检测模型
detect_model = YOLO("./weights/yolo11x.pt")
# 加载分类模型
# model = YOLO("yolo11n-cls.pt")  # load an official model
class_model = YOLO("./runs/classify/train_play_sleep_normal_unknown4+/weights/best.pt")

video_path = './test_img/test_video.mp4'
output_video_path = './test_img/result.mp4'

cam = cv2.VideoCapture(video_path)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cam.get(cv2.CAP_PROP_FPS))

# 创建 VideoWriter 对象，用于保存灰度视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=True)

while cam.isOpened():
    ret, frame = cam.read()
    img_h, img_w, _ = frame.shape

    # 检测模型预测行人类别
    detect_results = detect_model(frame, classes=[0])
    # 获取行人检测框坐标
    boxes = detect_results[0].boxes.xywh.cuda()
    for box in boxes:
        x, y, box_w, box_h = box  # x,y center point
        xmin = int(x) - int(box_w / 2) - offset
        ymin = int(y) - int(box_h / 2) - offset
        xmax = int(x) + int(box_w / 2) + offset
        ymax = int(y) + int(box_h / 2) + offset
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > img_w:
            xmax = img_w
        if ymax > img_h:
            ymax = img_h
        # print(xmin, ymin, xmax, ymax)

        crop_img = frame[ymin:ymax, xmin:xmax]
        resize_crop_img = cv2.resize(crop_img, (320, 320))

        # 分类模型对检测到的行人框进行预测
        class_results = class_model(resize_crop_img)
        # {0: 'normal', 1: 'play', 2: 'sleep', 3: 'unknown'}
        print(type(class_results[0].probs.top5[0]))
        if class_results[0].probs.top5[0] == 0:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, "normal", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        elif class_results[0].probs.top5[0] == 1:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
            cv2.putText(frame, "play", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        elif class_results[0].probs.top5[0] == 2:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)
            cv2.putText(frame, "sleep", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        else:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
            cv2.putText(frame, "unknow", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        out.write(frame)
        cv2.imshow("results", frame)

        # 最有可能的类别
        # print(class_results[0].names[0])
        # cls_frame = class_results[0].plot()
        # 可视化结果
        # cv2.imshow("results", cls_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cam.release()
cv2.destroyAllWindows()