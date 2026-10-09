"""Step 3 - live face recognition and attendance tracking.

Run this file directly for a live preview (names + confidence on the camera
feed, ESC to stop). main.py uses run_monitoring() for timed attendance sessions.

Faces:  camera frame -> YuNet face detector (Haar fallback) -> crop ->
        PyTorch CNN (or OpenCV LBPH if no CNN is trained) -> per-face smoothing
        -> presence timers -> attendance report.
Signs:  camera frame -> MediaPipe hand landmarks -> PyTorch network (trained on
        the public HaGRID gesture dataset) -> recognised word, shown in a side
        panel (and spoken aloud on macOS) so people who cannot speak can be understood.
"""
import json
import os
import subprocess
import sys
import time
from collections import defaultdict, deque

import cv2
import numpy as np


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

SIGN_FILE = os.path.join(MODELS_DIR, "sign_net.pt")

YUNET_FILE = os.path.join(ASSETS_DIR, "face_detection_yunet_2023mar.onnx")
HAND_FILE = os.path.join(ASSETS_DIR, "hand_landmarker.task")
HAAR_FILE = os.path.join(ASSETS_DIR, "haarcascade_frontalface_default.xml")

CNN_THRESHOLD = 0.7        # minimum confidence (70%) to accept a CNN match
LBPH_THRESHOLD = 70        # maximum LBPH distance to accept a match
PRESENT_FRACTION = 0.8     # present if seen for >= 80% of the session
BACKGROUND_PREFIX = "_background"

SIGN_FEATURES = 63         # 21 hand landmarks x (x, y, z), relative to the wrist
NO_SIGN_LABEL = "no_gesture"   # hand visible but not signing
SIGN_THRESHOLD = 0.75      # minimum (smoothed) confidence to accept a sign

# What each HaGRID gesture means on screen. Edit the words to suit your signs.
SIGN_WORDS = {
    "palm": "Hi / Hello",
    "stop": "Stop",
    "stop_inverted": "Stop",
    "like": "Good",
    "dislike": "Bad",
    "ok": "OK",
    "fist": "Yes",
    "peace": "Peace",
    "peace_inverted": "Peace",
    "call": "Call me",
    "mute": "Quiet",
    "rock": "Rock on",
    "one": "One",
    "two_up": "Two",
    "two_up_inverted": "Two",
    "three": "Three",
    "three2": "Three",
    "four": "Four",
}
PANEL_WIDTH = 380


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
        return name, f"{prob.item() * 100:.0f}%"


# =========================================================
# Hand tracking + sign recognition
# =========================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
    (18, 19), (19, 20), (0, 17),
]


def hand_features(landmarks, side):
    """21 landmarks -> 63 numbers, independent of hand size, position and left/right."""
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], np.float32)
    rel = pts - pts[0]  # relative to the wrist
    rel /= max(float(np.linalg.norm(rel[9, :2])), 1e-6)  # wrist -> middle-finger base
    if side == "Left":
        rel[:, 0] *= -1  # treat a left hand as a mirrored right hand
    return rel.ravel().astype(np.float32)


class HandTracker:
    """MediaPipe hand landmarker (current Tasks API).

    static=False: for a live video stream; static=True: for single images.
    process() -> (63 features, 21 (x, y) points in 0-1) or (None, None) if no hand.
    """

    def __init__(self, static=False):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.static = static
        mode = vision.RunningMode.IMAGE if static else vision.RunningMode.VIDEO
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_FILE),
            num_hands=1, running_mode=mode)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.start = time.time()
        self.last_ms = -1

    def process(self, frame):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if self.static:
            result = self.landmarker.detect(image)
        else:
            ms = max(int((time.time() - self.start) * 1000), self.last_ms + 1)
            self.last_ms = ms
            result = self.landmarker.detect_for_video(image, ms)

        if not result.hand_landmarks:
            return None, None

        landmarks = result.hand_landmarks[0]
        side = result.handedness[0][0].category_name
        return hand_features(landmarks, side), [(p.x, p.y) for p in landmarks]


def draw_hand(frame, points):
    if not points:
        return
    h, w = frame.shape[:2]
    px = [(int(x * w), int(y * h)) for x, y in points]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], (255, 200, 0), 2)
    for pt in px:
        cv2.circle(frame, pt, 4, (0, 140, 255), -1)


def pretty(word):
    return word.replace("_", " ").title()


def say_aloud(text):
    """Speaks without blocking the video loop (macOS `say`; silently skipped elsewhere)."""
    if sys.platform == "darwin":
        try:
            subprocess.Popen(["say", text])
        except OSError:
            pass


class SignRecognizer:
    """Turns a stream of per-frame hand features into recognised words."""

    def __init__(self, speak=True):
        import torch
        from face_train import SignNet

        self.torch = torch
        checkpoint = torch.load(SIGN_FILE, map_location="cpu")
        self.classes = checkpoint["classes"]
        self.net = SignNet(len(self.classes))
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()

        self.speak = speak
        self.recent = deque(maxlen=8)    # last few probability vectors (smoothing)
        self.history = deque(maxlen=7)   # recognised words, newest last
        self.current = ""                # what is being seen right now
        self.last_label = None
        self.armed = True                # False after a word, until the hand rests/changes
        self.candidate, self.streak, self.rest_run = None, 0, 0
        self.hand_visible = False

    @staticmethod
    def available():
        return os.path.exists(SIGN_FILE) and os.path.exists(HAND_FILE)

    def update(self, features):
        """Feed one frame (features or None). Returns a word when a new sign is recognised."""
        self.hand_visible = features is not None

        if features is None:  # hand left the picture: the next sign counts as new
            self.recent.clear()
            self.current, self.candidate, self.streak, self.armed = "", None, 0, True
            return None

        with self.torch.no_grad():
            logits = self.net(self.torch.from_numpy(features).unsqueeze(0))
            self.recent.append(self.torch.softmax(logits, dim=1)[0])
        average = self.torch.stack(list(self.recent)).mean(dim=0)
        prob, idx = average.max(0)
        label, prob = self.classes[idx.item()], prob.item()

        if label == NO_SIGN_LABEL or prob < SIGN_THRESHOLD:
            self.current, self.candidate, self.streak = "", None, 0
            self.rest_run = self.rest_run + 1 if label == NO_SIGN_LABEL else 0
            if self.rest_run >= 8:  # hand clearly rested -> the same word may be said again
                self.armed = True
            return None

        self.rest_run = 0
        word = SIGN_WORDS.get(label, pretty(label))
        self.current = f"{word} {prob * 100:.0f}%"
        self.streak = self.streak + 1 if label == self.candidate else 1
        self.candidate = label

        if self.streak >= 5 and (self.armed or label != self.last_label):
            self.last_label, self.armed = label, False
            self.history.append(word)
            if self.speak:
                say_aloud(word)
            return word
        return None


def draw_sign_panel(frame, gestures):
    """Returns the frame with a side panel showing what the person is signing."""
    h = frame.shape[0]
    panel = np.full((h, PANEL_WIDTH, 3), 30, np.uint8)

    cv2.putText(panel, "SIGN -> TEXT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.line(panel, (20, 55), (PANEL_WIDTH - 20, 55), (90, 90, 90), 1)

    latest = gestures.history[-1] if gestures.history else "..."
    cv2.putText(panel, latest, (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                1.6 if len(latest) < 9 else 1.0, (0, 255, 0), 3)

    status = gestures.current or ("Hand detected" if gestures.hand_visible else "Show your hand")
    cv2.putText(panel, status, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    cv2.putText(panel, "Said so far:", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1)
    y = 250
    for word in reversed(list(gestures.history)):
        cv2.putText(panel, word, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 36

    return np.hstack([frame, panel])


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
                   speak=None, signs=True):
    """Runs a timed session and writes the attendance report.

    source: camera index or video file; report_file=None skips saving the report.
    signs=True also recognises hand signs (when trained) and shows them in a side panel.
    Returns (report, presence_time).
    """
    recognizer = Recognizer()
    detector = FaceDetector()
    tracker = PresenceTracker()

    hands = gestures = None
    if signs and SignRecognizer.available():
        try:
            hands, gestures = HandTracker(), SignRecognizer()
        except Exception as error:  # e.g. PyTorch / MediaPipe problem -> faces only
            print(f"[WARNING] Sign recognition disabled: {error}")

    print(f"\n[INFO] Recognizer: {recognizer.name} | Detector: {detector.name} | "
          f"Signs: {'on' if gestures else 'off (run face_train.py)'}")
    print("[INFO] Monitoring started (press ESC in the camera window to stop)\n")

    cam = cv2.VideoCapture(source)
    start = time.time()

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if gestures:
            features, points = hands.process(frame)
            gestures.update(features)
            if show:
                draw_hand(frame, points)

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

            if gestures:
                frame = draw_sign_panel(frame, gestures)

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
