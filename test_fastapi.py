import requests
import time

def test_crack_segment():
    # 设置请求的 URL 和参数
    url = "http://127.0.0.1:8088/crack_segment"
    input_dir = r"/Users/hongliang/Desktop/yolov12/input"  # 替换为实际的文件夹路径
    output_dir = r"/Users/hongliang/Desktop/yolov12/output"
    filename = r"123.jpg"  # 替换为实际的文件名

    # 发送 POST 请求
    start_time = time.time()
    response = requests.post(url, params={"input_dir": input_dir, "output_dir": output_dir, "filename": filename})
    end_time = time.time()
    print(f"请求时间: {end_time - start_time}秒")

    # 打印响应状态码和内容
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")


def test_crack_segment_url():
    # 设置请求的 URL 和参数
    url = "http://127.0.0.1:8088/crack_segment_url"
    url_img = r"https://copyright.bdstatic.com/vcg/creative/7ecbfecab42ebbe694404a3f8af4fdd9.jpg"  # 替换为实际的文件夹路径
    output_dir = r"/Users/hongliang/Desktop/yolov12/output"

    # 发送 POST 请求
    start_time = time.time()
    response = requests.post(url, params={"url_img": url_img, "output_dir": output_dir})
    end_time = time.time()
    print(f"请求时间: {end_time - start_time}秒")

    # 打印响应状态码和内容
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")

# 运行测试函数
# test_crack_segment()
test_crack_segment_url()
