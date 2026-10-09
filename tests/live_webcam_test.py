"""Live webcam check: how often do the CNN and LBPH recognise the enrolled person?

    python tests/live_webcam_test.py [expected_name] [seconds]
"""
import json
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cnn_recognizer import CNNFaceRecognizer

expected = sys.argv[1] if len(sys.argv) > 1 else "Tarun"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 12

cnn = CNNFaceRecognizer()
lbph = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
lbph.read(os.path.join(ROOT, "models", "trainer_face.yml"))
lbph_labels = {int(k): v for k, v in json.load(open(os.path.join(ROOT, "models", "label_map.json"))).items()}
cascade = cv2.CascadeClassifier(os.path.join(ROOT, "assets", "haarcascade_frontalface_default.xml"))

cam = cv2.VideoCapture(0)
time.sleep(2)
frames = detected = 0
cnn_names, lbph_names, cnn_probs, lbph_conf = Counter(), Counter(), [], []
t0 = time.time()
while time.time() - t0 < seconds:
    ok, frame = cam.read()
    if not ok:
        break
    frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))
    if not len(boxes):
        continue
    detected += 1
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    face = gray[y:y + h, x:x + w]

    name, prob = cnn.predict(face)
    cnn_names[name] += 1; cnn_probs.append(prob)

    f220 = cv2.equalizeHist(cv2.resize(face, (220, 220)))
    id_, conf = lbph.predict(f220)
    lbph_names[lbph_labels.get(id_, "Unknown") if conf < 70 else "Unknown"] += 1
    lbph_conf.append(conf)
cam.release()

print(f"\nframes={frames}  face detected in {detected} ({detected / max(frames, 1):.0%})")
if detected:
    print(f"CNN  : {cnn_names.most_common()}  -> '{expected}' {cnn_names[expected] / detected:.0%}, mean prob {np.mean(cnn_probs):.2f}")
    print(f"LBPH : {lbph_names.most_common()}  -> '{expected}' {lbph_names[expected] / detected:.0%}, mean distance {np.mean(lbph_conf):.1f} (lower is better, cutoff 70)")
