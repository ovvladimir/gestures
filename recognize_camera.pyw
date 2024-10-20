# https://docs.ultralytics.com/usage/cli/#how-do-i-customize-yolo11-cli-commands-to-override-default-arguments

from imutils.video import FPS
from imutils.video import VideoStream
import time
import cv2
import os
from ultralytics import YOLO

args = {"model": "yolov8n"}
model = args["model"].lower()

dir = os.path.dirname(__file__)
modelPath1 = os.path.join(dir, "yolov8n/weights/best.pt")
# modelPath2 = os.path.join(dir, "yolov8n/weights/last.pt")
# modelPath1 = os.path.join(dir, "yolov8n/yolov8n")
# modelPath1 = os.path.join(dir, "yolov8s/weights/best.pt")
# detector = YOLO(modelPath1, modelPath2)
detector = YOLO(modelPath1)

vs = VideoStream(src=0).start()

fps = FPS().start()
counter = 0
color = (255, 255, 255)
startTime = time.monotonic()
time.sleep(1.0)

print("[INFO] starting camera...")
while True:
    frame = vs.read()

    # detect = detector.predict(frame, verbose=False)[0].verbose()[1:].replace(")", "").replace(",", "")
    for detect in detector.predict(frame, verbose=False):
        detect = detect.verbose().replace("(", "").replace(")", "").replace(",", "")
    cv2.putText(frame, detect, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, color)

    cv2.putText(frame, f"fps: {int(counter / (time.monotonic() - startTime))}",
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, color)

    fps.update()
    counter += 1

    cv2.imshow(model, frame)

    if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty(model, cv2.WND_PROP_VISIBLE) < 1:
        break

fps.stop()
print("[INFO] time: {:.2f}".format(fps.elapsed()))
print("[INFO] FPS: {:.2f}".format(fps.fps()))
vs.stop()
cv2.destroyAllWindows()
