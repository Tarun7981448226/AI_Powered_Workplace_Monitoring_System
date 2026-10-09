import json
import os
import time
from collections import defaultdict, deque

import cv2

from detector import FaceDetector

try:
    from cnn_recognizer import CNNFaceRecognizer
except ImportError:  # PyTorch not installed -> LBPH only
    CNNFaceRecognizer = None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.environ.get("FACE_MODELS_DIR", os.path.join(BASE_DIR, "models"))

TRAINER_FILE = os.path.join(MODELS_DIR, "trainer_face.yml")
LABEL_MAP_FILE = os.path.join(MODELS_DIR, "label_map.json")
ATTENDANCE_REPORT_FILE = os.path.join(DATA_DIR, "attendance_report.json")

PRESENT_FRACTION = 0.8
LBPH_THRESHOLD = 70
BACKGROUND_PREFIX = "_background"


# -----------------------------------------
# Recognition backend (CNN if trained, else LBPH)
# -----------------------------------------
class Recognizer:
    """predict(gray_face) -> name or 'Unknown', using the best available model."""

    def __init__(self):
        self.cnn = None
        self.lbph = None
        self.label_map = {}

        if CNNFaceRecognizer is not None and CNNFaceRecognizer.available():
            self.cnn = CNNFaceRecognizer()
            self.label_map = self.cnn.label_map
            self.name = "PyTorch CNN"
        elif os.path.exists(TRAINER_FILE) and os.path.exists(LABEL_MAP_FILE):
            self.lbph = cv2.face.LBPHFaceRecognizer_create()
            self.lbph.read(TRAINER_FILE)
            with open(LABEL_MAP_FILE) as f:
                self.label_map = {int(k): v for k, v in json.load(f).items()}
            self.name = "OpenCV LBPH"
        else:
            raise FileNotFoundError(
                "No trained model found. Run src/face_train.py (and optionally "
                "src/train_cnn.py) first.")

    def enrolled(self):
        return sorted(n for n in self.label_map.values()
                      if not n.startswith(BACKGROUND_PREFIX))

    def predict(self, gray_face):
        """Returns (name, detail); detail is the confidence shown on screen."""
        if self.cnn:
            name, prob = self.cnn.predict(gray_face)
            return name, f"{prob:.2f}"

        face = cv2.equalizeHist(cv2.resize(gray_face, (220, 220)))
        id_, distance = self.lbph.predict(face)
        detail = f"d={distance:.0f}"
        if distance < LBPH_THRESHOLD:
            return self.label_map.get(id_, "Unknown"), detail
        return "Unknown", detail


# -----------------------------------------
# Presence tracking + attendance report
# -----------------------------------------
class PresenceTracker:
    TRACK_TIMEOUT = 1.0  # seconds without a match before a face track is dropped

    def __init__(self, smoothing_window=5, min_votes=3):
        self.presence_time = defaultdict(float)
        self.last_seen = {}
        self.tracks = []  # each: {"center", "size", "votes", "t"}
        self.window = smoothing_window
        self.min_votes = min_votes

    def stable_name(self, box, name, now=None):
        """Majority vote over recent frames of the *same face* (matched by position).

        Faces are matched to the nearest previous box, so a person who moves or
        whose detection box jitters keeps their votes.
        """
        now = time.time() if now is None else now
        x, y, w, h = box
        center = (x + w / 2, y + h / 2)

        self.tracks = [t for t in self.tracks if now - t["t"] < self.TRACK_TIMEOUT]

        best, best_dist = None, None
        for t in self.tracks:
            dist = ((t["center"][0] - center[0]) ** 2 + (t["center"][1] - center[1]) ** 2) ** 0.5
            if dist < 0.75 * max(w, h, t["size"]) and (best is None or dist < best_dist):
                best, best_dist = t, dist

        if best is None:
            best = {"votes": deque(maxlen=self.window)}
            self.tracks.append(best)
        best.update(center=center, size=max(w, h), t=now)

        votes = best["votes"]
        votes.append(name)
        winner = max(set(votes), key=votes.count)
        return winner if votes.count(winner) >= self.min_votes else "Unknown"

    def update(self, name, now):
        if name == "Unknown" or name.startswith(BACKGROUND_PREFIX):
            return
        if name not in self.last_seen:
            self.last_seen[name] = now
        elapsed = now - self.last_seen[name]
        if elapsed < 2:
            self.presence_time[name] += elapsed
        self.last_seen[name] = now


def compute_report(presence_time, total_seconds, enrolled=(), fraction=PRESENT_FRACTION):
    """Present if seen for >= fraction of the session; enrolled people never seen are Absent."""
    required = total_seconds * fraction
    report = {}
    for person in enrolled:
        report[person] = "Absent"
    for person, seconds in presence_time.items():
        report[person] = "Present" if seconds >= required else "Absent"
    return report


# -----------------------------------------
# Monitoring session
# -----------------------------------------
def run_monitoring(total_seconds, source=0, show=True, report_file=ATTENDANCE_REPORT_FILE,
                   speak=None, recognizer=None, detector=None):
    """Runs a monitoring session and writes the attendance report.

    source: camera index or a video file path. Returns (report, presence_time).
    """
    say = speak or (lambda text: None)
    recognizer = recognizer or Recognizer()
    detector = detector or FaceDetector()
    tracker = PresenceTracker()

    print(f"\n[INFO] Recognizer: {recognizer.name} | Detector: {detector.name}")
    print("[INFO] Monitoring started (press ESC in the camera window to stop)\n")

    cam = cv2.VideoCapture(source)
    start = time.time()
    frames = 0

    while True:
        ok, frame = cam.read()
        if not ok:
            break
        frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now = time.time()

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

    elapsed = max(time.time() - start, 1e-6)
    report = compute_report(tracker.presence_time, min(total_seconds, elapsed),
                            recognizer.enrolled())

    print("\n===== FINAL REPORT =====\n")
    for person, status in sorted(report.items()):
        print(f"{person} -> {int(tracker.presence_time.get(person, 0))}s -> {status}")
    print(f"\n[INFO] {frames} frames processed")

    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)

    say("Monitoring finished report generated")
    return report, dict(tracker.presence_time)
