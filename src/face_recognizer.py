"""Step 3 - live face recognition and attendance tracking.

Run this file directly for a live preview (names + confidence on the camera
feed, ESC to stop). main.py uses run_monitoring() for timed attendance sessions.

Pipeline: camera frame -> YuNet face detector (Haar fallback) -> crop ->
PyTorch CNN (or OpenCV LBPH if no CNN is trained) -> per-face smoothing ->
presence timers -> attendance report.
"""
import json
import os
import time
from collections import defaultdict, deque

import cv2


# -----------------------------
# Paths / settings
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

TRAINER_FILE = os.path.join(MODELS_DIR, "trainer_face.yml")
LABEL_MAP_FILE = os.path.join(MODELS_DIR, "label_map.json")
CNN_FILE = os.path.join(MODELS_DIR, "face_cnn.pt")
CNN_LABEL_FILE = os.path.join(MODELS_DIR, "label_map_cnn.json")
ATTENDANCE_REPORT_FILE = os.path.join(DATA_DIR, "attendance_report.json")

YUNET_FILE = os.path.join(ASSETS_DIR, "face_detection_yunet_2023mar.onnx")
HAAR_FILE = os.path.join(ASSETS_DIR, "haarcascade_frontalface_default.xml")

CNN_THRESHOLD = 0.7        # minimum softmax probability to accept a CNN match
LBPH_THRESHOLD = 70        # maximum LBPH distance to accept a match
PRESENT_FRACTION = 0.8     # present if seen for >= 80% of the session
BACKGROUND_PREFIX = "_background"


# =========================================================
# Face detection
# =========================================================
class FaceDetector:
    """YuNet (pretrained neural network) with a Haar cascade fallback.

    detect(frame) -> list of (x, y, w, h) boxes inside the frame, largest first.
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
        height, width = frame.shape[:2]

        if self.yunet is not None:
            bgr = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            self.yunet.setInputSize((width, height))
            _, found = self.yunet.detect(bgr)
            boxes = [] if found is None else [tuple(int(v) for v in f[:4]) for f in found]
        else:
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found = self.haar.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5,
                minSize=(self.min_size, self.min_size))
            boxes = [tuple(int(v) for v in b) for b in found]

        clipped = []
        for x, y, w, h in boxes:
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + w, width), min(y + h, height)
            if x1 - x0 >= self.min_size and y1 - y0 >= self.min_size:
                clipped.append((x0, y0, x1 - x0, y1 - y0))

        return sorted(clipped, key=lambda b: b[2] * b[3], reverse=True)


# =========================================================
# Recognition (CNN if trained, else LBPH)
# =========================================================
class Recognizer:
    """predict(gray_face) -> (name, detail). Name is 'Unknown' if not confident."""

    def __init__(self):
        self.cnn = self.lbph = None
        self.label_map = {}

        if os.path.exists(CNN_FILE) and os.path.exists(CNN_LABEL_FILE) and self._load_cnn():
            self.name = "PyTorch CNN"
        elif os.path.exists(TRAINER_FILE) and os.path.exists(LABEL_MAP_FILE):
            self.lbph = cv2.face.LBPHFaceRecognizer_create()
            self.lbph.read(TRAINER_FILE)
            with open(LABEL_MAP_FILE) as f:
                self.label_map = {int(k): v for k, v in json.load(f).items()}
            self.name = "OpenCV LBPH"
        else:
            raise FileNotFoundError("No trained model found. Run face_train.py first.")

    def _load_cnn(self):
        try:
            import torch
            from face_train import FaceCNN, get_device
        except ImportError:
            return False  # PyTorch not installed -> use LBPH

        checkpoint = torch.load(CNN_FILE, map_location="cpu")
        self.device = get_device()
        self.cnn = FaceCNN(checkpoint["num_classes"])
        self.cnn.load_state_dict(checkpoint["state_dict"])
        self.cnn.to(self.device).eval()

        with open(CNN_LABEL_FILE) as f:
            self.label_map = {int(k): v for k, v in json.load(f).items()}
        return True

    def enrolled(self):
        return sorted(n for n in self.label_map.values() if not n.startswith(BACKGROUND_PREFIX))

    def predict(self, gray_face):
        if self.cnn is not None:
            return self._predict_cnn(gray_face)

        face = cv2.equalizeHist(cv2.resize(gray_face, (220, 220)))
        id_, distance = self.lbph.predict(face)
        name = self.label_map.get(id_, "Unknown") if distance < LBPH_THRESHOLD else "Unknown"
        return name, f"d={distance:.0f}"

    def _predict_cnn(self, gray_face):
        import torch
        import torch.nn.functional as F
        from face_train import IMG_SIZE, preprocess_face

        face = preprocess_face(gray_face)
        x = torch.from_numpy(face).float().div(255).sub(0.5).div(0.5)
        x = x.view(1, 1, IMG_SIZE, IMG_SIZE).to(self.device)

        with torch.no_grad():
            probs = F.softmax(self.cnn(x), dim=1)[0]
        prob, idx = probs.max(0)

        name = self.label_map.get(idx.item(), "Unknown")
        if prob.item() < CNN_THRESHOLD or name.startswith(BACKGROUND_PREFIX):
            name = "Unknown"
        return name, f"{prob.item():.2f}"


# =========================================================
# Presence tracking + attendance report
# =========================================================
class PresenceTracker:
    """Smooths predictions per face and accumulates presence time per person."""

    TRACK_TIMEOUT = 1.0  # seconds without a match before a face track is dropped

    def __init__(self, window=5, min_votes=3):
        self.presence_time = defaultdict(float)
        self.last_seen = {}
        self.tracks = []
        self.window = window
        self.min_votes = min_votes

    def stable_name(self, box, name, now):
        """Majority vote over recent frames of the same face (matched by position)."""
        x, y, w, h = box
        center = (x + w / 2, y + h / 2)

        self.tracks = [t for t in self.tracks if now - t["time"] < self.TRACK_TIMEOUT]

        track, best = None, None
        for t in self.tracks:
            dist = ((t["center"][0] - center[0]) ** 2 + (t["center"][1] - center[1]) ** 2) ** 0.5
            if dist < 0.75 * max(w, h, t["size"]) and (best is None or dist < best):
                track, best = t, dist

        if track is None:
            track = {"votes": deque(maxlen=self.window)}
            self.tracks.append(track)
        track.update(center=center, size=max(w, h), time=now)

        track["votes"].append(name)
        winner = max(set(track["votes"]), key=track["votes"].count)
        return winner if track["votes"].count(winner) >= self.min_votes else "Unknown"

    def update(self, name, now):
        if name == "Unknown" or name.startswith(BACKGROUND_PREFIX):
            return

        if name not in self.last_seen:
            self.last_seen[name] = now

        elapsed = now - self.last_seen[name]
        if elapsed < 2:
            self.presence_time[name] += elapsed
        self.last_seen[name] = now


def compute_report(presence_time, total_seconds, enrolled=()):
    """Present if seen for >= 80% of the session; enrolled people never seen are Absent."""
    report = {person: "Absent" for person in enrolled}
    for person, seconds in presence_time.items():
        report[person] = "Present" if seconds >= total_seconds * PRESENT_FRACTION else "Absent"
    return report


# =========================================================
# Monitoring session
# =========================================================
def run_monitoring(total_seconds, source=0, show=True, report_file=ATTENDANCE_REPORT_FILE,
                   speak=None):
    """Runs a timed session and writes the attendance report.

    source: camera index or video file; report_file=None skips saving the report.
    Returns (report, presence_time).
    """
    recognizer = Recognizer()
    detector = FaceDetector()
    tracker = PresenceTracker()

    print(f"\n[INFO] Recognizer: {recognizer.name} | Detector: {detector.name}")
    print("[INFO] Monitoring started (press ESC in the camera window to stop)\n")

    cam = cv2.VideoCapture(source)
    start = time.time()

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for box in detector.detect(frame):
            x, y, w, h = box
            raw_name, detail = recognizer.predict(gray[y:y + h, x:x + w])
            name = tracker.stable_name(box, raw_name, now)
            tracker.update(name, now)

            if show:
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{name} ({detail})", (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if show:
            line_y = 30
            for person, seconds in tracker.presence_time.items():
                cv2.putText(frame, f"{person}: {int(seconds)}s", (10, line_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                line_y += 30

            cv2.imshow("AI Monitoring System", frame)
            if cv2.waitKey(1) == 27:
                break

        if now - start >= total_seconds:
            break

    cam.release()
    if show:
        cv2.destroyAllWindows()

    duration = min(total_seconds, max(time.time() - start, 1e-6))
    report = compute_report(tracker.presence_time, duration, recognizer.enrolled())

    print("\n===== FINAL REPORT =====\n")
    for person, status in sorted(report.items()):
        print(f"{person} -> {int(tracker.presence_time.get(person, 0))}s -> {status}")

    if report_file:
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, "w") as f:
            json.dump(report, f, indent=4)

    if speak:
        speak("Monitoring finished report generated")

    return report, dict(tracker.presence_time)


# =========================================================
# MAIN - live preview
# =========================================================
if __name__ == "__main__":
    try:
        run_monitoring(total_seconds=24 * 3600, show=True, report_file=None)
    except FileNotFoundError as error:
        print("[ERROR]", error)
