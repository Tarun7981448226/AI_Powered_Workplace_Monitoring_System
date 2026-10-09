import os

import cv2


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

YUNET_FILE = os.path.join(ASSETS_DIR, "face_detection_yunet_2023mar.onnx")
HAAR_FILE = os.path.join(ASSETS_DIR, "haarcascade_frontalface_default.xml")


class FaceDetector:
    """Face detector: YuNet (pretrained neural network) with a Haar fallback.

    detect(frame) takes a BGR (or grayscale) frame and returns a list of
    (x, y, w, h) boxes inside the frame, largest first.
    """

    def __init__(self, min_size=80, score_threshold=0.6):
        self.min_size = min_size
        self.yunet = None

        if os.path.exists(YUNET_FILE) and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    YUNET_FILE, "", (320, 320), score_threshold, 0.3, 5000)
            except cv2.error:
                self.yunet = None

        self.haar = cv2.CascadeClassifier(HAAR_FILE)
        self.name = "YuNet" if self.yunet is not None else "Haar"

    def detect(self, frame):
        h_img, w_img = frame.shape[:2]
        boxes = []

        if self.yunet is not None:
            bgr = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            self.yunet.setInputSize((w_img, h_img))
            _, faces = self.yunet.detect(bgr)
            if faces is not None:
                boxes = [tuple(int(v) for v in f[:4]) for f in faces]
        else:
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found = self.haar.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5,
                minSize=(self.min_size, self.min_size))
            boxes = [tuple(int(v) for v in b) for b in found]

        clipped = []
        for x, y, w, h in boxes:
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + w, w_img), min(y + h, h_img)
            if x1 - x0 >= self.min_size and y1 - y0 >= self.min_size:
                clipped.append((x0, y0, x1 - x0, y1 - y0))

        return sorted(clipped, key=lambda b: b[2] * b[3], reverse=True)
