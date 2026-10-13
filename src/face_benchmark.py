"""Step 6 (optional) - latency and CPU benchmark of the face pipeline.

Builds test frames from FairFace validation faces (run face_fairness.py once
first so the data is downloaded) pasted onto a 640x480 or 1280x720 background,
then times three pipelines on exactly the same frames:

  1. v1         - the first version of this project: Haar cascade
                  (scaleFactor 1.2, minNeighbors 5, minSize 100) + LBPH
  2. full-res   - YuNet + PyTorch CNN, detecting on the full frame
  3. optimized  - YuNet + PyTorch CNN, detecting on a copy at most 640 px wide
                  (what face_recognizer.py does now)

For each it reports latency per frame, CPU time per frame and how often the
face was found. Needs trained models (face_train.py). Output:
data/benchmark_report.json.
"""
import json
import os
import time

import cv2
import numpy as np

from face_recognizer import (FaceDetector, HAAR_FILE, LABEL_MAP_FILE, LBPH_THRESHOLD,
                             Recognizer, TRAINER_FILE, DATA_DIR)
from face_fairness import FAIRFACE_DIR, PARQUET_FILES

REPORT_FILE = os.path.join(DATA_DIR, "benchmark_report.json")
RESOLUTIONS = [(640, 480), (1280, 720)]
N_FRAMES = 60
REPEATS = 3


def fairface_faces(n):
    import pyarrow.parquet as pq

    table = pq.read_table(os.path.join(FAIRFACE_DIR, PARQUET_FILES["val"][0]), columns=["image"])
    images = table.column("image").to_pylist()[:n]
    return [cv2.imdecode(np.frombuffer(img["bytes"], np.uint8), cv2.IMREAD_COLOR) for img in images]


def make_frames(faces, size, seed=0):
    """One face per frame, at a random position and a webcam-like size."""
    rng = np.random.default_rng(seed)
    w, h = size
    frames, boxes = [], []
    for face in faces:
        noise = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
        frame = cv2.GaussianBlur(cv2.resize(noise, (w, h)), (0, 0), 9)
        side = int(h * rng.uniform(0.45, 0.65))
        x, y = int(rng.integers(0, w - side)), int(rng.integers(0, h - side))
        frame[y:y + side, x:x + side] = cv2.resize(face, (side, side))
        frames.append(frame)
        boxes.append((x, y, side, side))
    return frames, boxes


class V1Pipeline:
    """The original detector and recognizer settings (first commit)."""
    name = "v1: Haar + LBPH"

    def __init__(self):
        self.haar = cv2.CascadeClassifier(HAAR_FILE)
        self.lbph = cv2.face.LBPHFaceRecognizer_create()
        self.lbph.read(TRAINER_FILE)
        with open(LABEL_MAP_FILE) as f:
            self.label_map = {int(k): v for k, v in json.load(f).items()}

    def __call__(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = self.haar.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
        for x, y, w, h in found:
            face = cv2.resize(gray[y:y + h, x:x + w], (220, 220))
            id_, distance = self.lbph.predict(face)
            _ = self.label_map.get(id_) if distance < LBPH_THRESHOLD else "Unknown"
        return len(found)


class CNNPipeline:
    def __init__(self, detect_width, recognizer):
        self.name = ("optimized: YuNet (640 px) + CNN" if detect_width
                     else "full-res: YuNet + CNN")
        self.detector = FaceDetector(detect_width=detect_width)
        self.recognizer = recognizer

    def __call__(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detect(frame)
        for x, y, w, h in boxes:
            self.recognizer.predict(gray[y:y + h, x:x + w])
        return len(boxes)


def benchmark(pipeline, frames):
    for frame in frames[:5]:  # warm-up (model loading, MPS kernels)
        pipeline(frame)
    times, found = [], 0
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    for _ in range(REPEATS):
        for frame in frames:
            t = time.perf_counter()
            found += pipeline(frame) > 0
            times.append(time.perf_counter() - t)
    cpu, wall = time.process_time() - cpu_start, time.perf_counter() - wall_start
    n = len(times)
    return {"median_ms": float(np.median(times) * 1000), "mean_ms": float(np.mean(times) * 1000),
            "cpu_ms_per_frame": cpu / n * 1000, "cpu_percent_of_one_core": cpu / wall * 100,
            "face_found_rate": found / n}


if __name__ == "__main__":
    cv2.setNumThreads(os.cpu_count() or 1)
    recognizer = Recognizer()
    pipelines = [V1Pipeline(), CNNPipeline(None, recognizer), CNNPipeline(640, recognizer)]
    faces = fairface_faces(N_FRAMES)

    report = {}
    for size in RESOLUTIONS:
        frames, _ = make_frames(faces, size)
        key = f"{size[0]}x{size[1]}"
        report[key] = {}
        print(f"\n[RESULT] {key}, {len(frames)} frames x {REPEATS}")
        print(f"   {'pipeline':34s}{'median ms':>11s}{'CPU ms':>9s}{'CPU %':>8s}{'found':>8s}")
        for p in pipelines:
            r = benchmark(p, frames)
            report[key][p.name] = r
            print(f"   {p.name:34s}{r['median_ms']:11.1f}{r['cpu_ms_per_frame']:9.1f}"
                  f"{r['cpu_percent_of_one_core']:7.0f}%{r['face_found_rate']:8.0%}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump({"frames": N_FRAMES, "repeats": REPEATS, "cpu_cores": os.cpu_count(),
                   "results": report}, f, indent=2)
    print(f"\n[INFO] Report saved: {REPORT_FILE}")
    print("       CPU % is process CPU time / wall time (100% = one core fully busy).")
