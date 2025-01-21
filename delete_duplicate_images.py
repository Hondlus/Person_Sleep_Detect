import os
import hashlib


def remove_duplicate_images(directory):
    files = os.listdir(directory)
    images = {}
    duplicate_image_names = []

    for file in files:
        # if file.endswith(".jpg") or file.endswith(".png"):
        with open(directory + '/' + file, 'rb') as f:
            content = f.read()
            hash_value = hashlib.md5(content).hexdigest()
            if hash_value in images:
                duplicate_image_names.append(file)
            else:
                images[hash_value] = file

    # print(duplicate_image_names)

    for img in duplicate_image_names:
        os.remove(directory + '/' + img)

    duplicate_image_names.clear()

if __name__ == '__main__':

    remove_duplicate_images("C:/Users/DXW/Desktop/Person_Sleep_Detect/download_img/单人职场员工睡觉")