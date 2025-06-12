import requests
import cv2
import numpy as np


def test_api_with_image_file(image_path):
    """
    测试API接口，使用本地图片文件
    :param image_path: 图片文件路径
    """
    url = "http://localhost:8086/predict"

    try:
        # 读取变成二进制图像，并发送到API
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
            response = requests.post(url, data=img_bytes)
            print(response.json())

        if response.status_code == 200:
            # 保存返回的图片
            output_path = "output_api_result.jpg"
            with open(output_path, 'wb') as f:
                f.write(bytes(response.json()["img_bytes"]))
            print(f"API测试成功，结果已保存到 {output_path}")

            # 显示图片
            img = cv2.imdecode(np.frombuffer(bytes(response.json()["img_bytes"]), np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow("API Result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"API请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")


# 使用示例
test_api_with_image_file("./input/1.jpg")  # 替换为你的测试图片路径
