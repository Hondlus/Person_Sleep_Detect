# from ultralytics import YOLO
# model = YOLO('./weights/yolo11x-cls.pt')
# results = model("https://ultralytics.com/images/bus.jpg")

from ultralytics import YOLO
import cv2

# Load a class model
model = YOLO('./weights/yolo11x-cls.pt')  # load an official model
# model = YOLO("./weights/yolo11n.pt")  # load an official model

# video_path = "rtsp://admin:HikFIATCT@192.168.50.11:554/Streaming/Channels/102"
# video_path = "rtsp://admin:Dxw202409@192.168.50.20:554/stream2"  # 15fps 640 480
# video_path = "rtsp://admin:HikNJQXFP@192.168.50.10:554/Streaming/Channels/102" # 屋内大屏摄像头
video_path = 0

cap = cv2.VideoCapture(video_path)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # Predict with the model
    results = model(frame, device=0)  # predict on an image

    frame = results[0].plot()

    cv2.imshow("Classing", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()