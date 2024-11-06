from threading import Thread
import cv2


class VideoStream:
    def __init__(self, src=0, name="Thread"):
        self.stream = cv2.VideoCapture(src)
        cv2.waitKey(100)
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.stream.read()
        self.name = name
        self.stopped = True

    def start(self):
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while self.stopped:
            self.ret, self.frame = self.stream.read()

    def video(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = False
