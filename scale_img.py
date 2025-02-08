import cv2
import os


scale = 640
img_path = './testimg/tingzhixian/'
save_path = './testimg/scale_img/'

for img_name in os.listdir(img_path):

    img = cv2.imread(img_path + img_name)
    height, width, _ = img.shape

    if height > width:
        scale_height = scale
        scale_width = int(width * scale / height)
        img = cv2.resize(img, (scale_width, scale_height))
        cv2.imwrite(save_path + img_name, img)
    else:
        scale_width = scale
        scale_height = int(height * scale / width)
        img = cv2.resize(img, (scale_width, scale_height))
        cv2.imwrite(save_path + img_name, img)
