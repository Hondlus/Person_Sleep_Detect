import os
from ultralytics import YOLO
import cv2

# 0: person
class_id = 0

# 注意：路径最后要带斜杠
img_path = "E:/person_sleep_detect/datasets/images/"
# 存储图片的路径
image_path = "E:/person_sleep_detect/datasets/images/sleep/"
image_path2 = "E:/person_sleep_detect/datasets/images/play/"
image_path3 = "E:/person_sleep_detect/datasets/images/normal/"
image_path4 = "E:/person_sleep_detect/datasets/images/unknown/"

# Load a model
model = YOLO("./weights/yolo11n.pt")  # load an official model

if __name__ == '__main__':
    count = 11111
    offset = 20 # 截取图片的偏移
    # 批量读取图片
    for img_name in os.listdir(img_path):
        img = cv2.imread(img_path + img_name)
        img_h, img_w, _ = img.shape
        print("img_h, img_w: ", img_h, img_w)

        # Predict with the model
        results = model(img, classes=[0], device=0)  # predict on an image
        # annotated_frame = results[0].plot()
        # cv2.imshow("Detecting", annotated_frame)

        boxes = results[0].boxes.xywh.cuda()
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
            print(xmin, ymin, xmax, ymax)

            crop_img = img[ymin:ymax, xmin:xmax]
            # 根据标注信息画框
            # cv2.imshow('crop_img', crop_img)
            resize_img = cv2.resize(crop_img, (320, 320))
            cv2.imshow('resize_crop_img', resize_img)
            # 按下s键保存到sleep文件夹
            if cv2.waitKey(0) & 0xff == ord("s"):
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                cv2.imwrite(image_path + str(count) + ".jpg", resize_img)
                count += 1

            # 按下d键保存到play文件夹
            if cv2.waitKey(0) & 0xff == ord("d"):
                if not os.path.exists(image_path2):
                    os.makedirs(image_path2)
                cv2.imwrite(image_path2 + str(count) + ".jpg", resize_img)
                count += 1

            # 按下f键保存到normal文件夹
            if cv2.waitKey(0) & 0xff == ord("f"):
                if not os.path.exists(image_path3):
                    os.makedirs(image_path3)
                cv2.imwrite(image_path3 + str(count) + ".jpg", resize_img)
                count += 1

            # 按下g键保存到unknown文件夹
            if cv2.waitKey(0) & 0xff == ord("g"):
                if not os.path.exists(image_path4):
                    os.makedirs(image_path4)
                cv2.imwrite(image_path4 + str(count) + ".jpg", resize_img)
                count += 1

            # 按下p键pass过去
            if cv2.waitKey(0) & 0xff == ord("p"):
                pass

        ## 按下q键退出程序
        if cv2.waitKey(0) & 0xff == ord("q"):
            break

        cv2.destroyAllWindows()