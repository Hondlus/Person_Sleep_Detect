from ultralytics import YOLO
import cv2
import os


def accuracy_test():

    correct_num = 0
    error_num = 0

    for folder_name in os.listdir(root):
        img_path = root + folder_name
        for img_name in os.listdir(img_path):
            img = cv2.imread(img_path + '/' + img_name)
            print('img_path', img_path + '/' + img_name)
            img_h, img_w, _ = img.shape

            # 检测模型预测行人类别
            detect_results = detect_model(img, classes=[0])
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

                crop_img = img[ymin:ymax, xmin:xmax]
                resize_crop_img = cv2.resize(crop_img, (320, 320))

                # 分类模型对检测到的行人框进行预测
                class_results = class_model(resize_crop_img)
                # {0: 'normal', 1: 'play', 2: 'sleep', 3: 'unknown'}
                # print("class leibie: {}".format(class_results[0].probs.top5[0]))

                if int(folder_name) == class_results[0].probs.top5[0]:
                    correct_num = correct_num + 1
                else:
                    error_num = error_num + 1

                if class_results[0].probs.top5[0] == 0:
                # if class_results[0].probs.top5[0] == int(img_name):
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(img, "normal", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                elif class_results[0].probs.top5[0] == 1:
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                    cv2.putText(img, "play", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                elif class_results[0].probs.top5[0] == 2:
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)
                    cv2.putText(img, "sleep", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                elif class_results[0].probs.top5[0] == 3:
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                    cv2.putText(img, "unknow", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                else:
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
                    cv2.putText(img, "else", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                resize_img = cv2.resize(img, (480, 480))
                cv2.imshow("results", resize_img)
                # cv2.imshow("results", img)

                # 最有可能的类别
                # print(class_results[0].names[0])
                # cls_frame = class_results[0].plot()
                # 可视化结果
                # cv2.imshow("results", cls_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()

    print('--------------------------------')
    accuracy_result = correct_num / (correct_num + error_num)
    print("accuracy: {}".format(accuracy_result))

if __name__ == '__main__':

    # 裁剪图片预留的空间填充
    offset = 20
    root = "./test_img/test_img/"

    # 加载检测模型
    detect_model = YOLO("./weights/yolo11n.pt")
    # 加载分类模型
    # model = YOLO("yolo11n-cls.pt")  # load an official model
    class_model = YOLO("./runs/classify/train_play_sleep_normal_unknown4+/weights/best.pt")
    # class_model = YOLO("./runs/classify/train/weights/best.pt")

    # 测试
    accuracy_test()