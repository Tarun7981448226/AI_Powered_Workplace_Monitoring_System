"""Step 1 - enroll a person: captures 30 face photos (look straight, then left, then right).

Hand signs do not need to be recorded: face_train.py learns them from a large
public gesture dataset (HaGRID). Run face_train.py after enrolling.
"""
import os
import time

import cv2
import pyttsx3

from face_recognizer import FaceDetector


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES_DIR = os.path.join(BASE_DIR, "dataset", "faces")


# -----------------------------
# Text to speech
# -----------------------------
try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
except Exception:  # no speech engine available
    engine = None


def speak(text):
    if engine is None:
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass


def open_camera(width=1280, height=720):
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    time.sleep(2)  # camera warm-up
    return cam


# =========================================================
# 1) Faces
# =========================================================
def capture_faces(directory):
    os.makedirs(directory, exist_ok=True)

    detector = FaceDetector(min_size=120)
    print("[INFO] Face detector:", detector.name)

    cam = open_camera()

    phases = [
        ("Look straight", 20),
        ("Turn head left slowly", 5),
        ("Turn head right slowly", 5),
    ]

    total = 0
    print("\n[INFO] Guided capture starting...")

    for instruction, limit in phases:

        speak(instruction)
        print("[INFO]", instruction)
        time.sleep(3)  # time to move head

        count = 0

        while count < limit:

            ok, frame = cam.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for (x, y, w, h) in detector.detect(frame)[:1]:

                face = cv2.equalizeHist(cv2.resize(gray[y:y + h, x:x + w], (220, 220)))

                total += 1
                count += 1
                cv2.imwrite(os.path.join(directory, f"{total}.jpg"), face)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, f"{instruction} ({count}/{limit})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Face Capture", frame)

            if cv2.waitKey(400) == 27:  # slow capture avoids duplicate photos
                break

    cam.release()
    cv2.destroyAllWindows()

    speak("Face capture complete")
    print("[INFO] Captured", total, "images.")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    while True:
        print("\n========== NEW USER ==========")
        name = input("Enter user name: ").strip()

        if not name:
            print("Name cannot be empty")
            continue

        capture_faces(os.path.join(FACES_DIR, name))
        print("\n[INFO] Face data saved!")

        if input("\nAdd another user? (y/n): ").lower() != "y":
            break

    print("\n[INFO] Done. Next: run face_train.py")
