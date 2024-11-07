# https://docs.ultralytics.com/usage/cli/#how-do-i-customize-yolo11-cli-commands-to-override-default-arguments
# https://universe.roboflow.com/lebanese-university-grkoz/hand-gesture-recognition-y5827/dataset/2/images/?split=test&predictions=true

from camvideostream import VideoStream
from collections import deque
import time
import cv2
import os
from ultralytics import YOLO

dir = os.path.dirname(__file__)
modelPath = os.path.join(dir, "yolov8n/best.pt")
detector = YOLO(modelPath)
# modelPath1 = os.path.join(dir, "yolov8n/best.pt")  # 50 epoch
# modelPath2 = os.path.join(dir, "yolov8s/best.pt")  # 50 epoch
# modelPath = os.path.join(dir, "yolo11n/yolo11n.pt")
# detector = YOLO(modelPath1, modelPath2)

vs = VideoStream().start()
model = os.path.basename(os.path.dirname(modelPath)).lower()
numframe = 0
color = (255, 255, 255)
timer_start = time.monotonic()
timestamps = deque([timer_start], maxlen=30)

print()
print("[INFO] starting camera...")


def fps() -> float:
    timestamps.append(time.monotonic())
    times = timestamps[-1] - timestamps[0]
    return round(len(timestamps) / times, 1)


while True:
    r, f = vs.video()
    if not r:
        break
    frame = f

    detect = detector.predict(frame, verbose=False)[0].verbose().replace("(", "").replace(")", "").replace(",", "")

    cv2.putText(frame, detect, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
    cv2.putText(frame, f"fps: {fps()}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, color)

    cv2.imshow(model, frame)

    if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty(model, cv2.WND_PROP_VISIBLE) < 1:
        break


print(f"[INFO] time: {round(time.monotonic() - timer_start, 2)}")
print(f"[INFO] FPS: {fps()}")
print("[INFO] stopped camera...")
vs.stop()
cv2.destroyAllWindows()
