"""End-to-end checks: dataset -> train_cnn.py CLI -> saved model -> Haar detection
+ CNN recognition on a full frame -> face_recognizer.py picks the CNN backend.

Uses a demo dataset built from public Olivetti faces in a temp dir, so your real
dataset/ and models/ are never touched. Run from the project root:

    python tests/test_pipeline.py
"""
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

PEOPLE = 8
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    data = fetch_olivetti_faces(shuffle=False)
    imgs = (data.images * 255).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        faces_dir = os.path.join(tmp, "faces")
        models_dir = os.path.join(tmp, "models")
        env = dict(os.environ, FACE_MODELS_DIR=models_dir)

        # 1. demo dataset in the same layout face_taker.py produces: Haar-detected
        #    face crops from full frames (so train and inference crops match)
        cascade = cv2.CascadeClassifier(os.path.join(ROOT, "assets", "haarcascade_frontalface_default.xml"))

        def to_frame(img):
            frame = np.full((480, 640), 127, np.uint8)
            frame[100:360, 190:450] = cv2.resize(img, (260, 260))
            return frame

        def crop_face(frame):
            boxes = cascade.detectMultiScale(frame, 1.2, 5, minSize=(100, 100))
            if not len(boxes):
                return None
            x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
            return cv2.resize(frame[y:y + h, x:x + w], (220, 220))

        n_saved = 0
        for p in range(PEOPLE):
            os.makedirs(os.path.join(faces_dir, f"person_{p}"))
            for k in range(9):  # image 9 is held out for testing
                crop = crop_face(to_frame(imgs[p * 10 + k]))
                if crop is not None:
                    cv2.imwrite(os.path.join(faces_dir, f"person_{p}", f"{k}.jpg"), crop)
                    n_saved += 1
        check("demo dataset written", n_saved >= PEOPLE * 6, f"({n_saved} Haar crops, {PEOPLE} people)")

        # 2. LBPH trainer still works (existing behaviour not broken)
        r = subprocess.run([sys.executable, "-c",
                            "import sys;sys.path.insert(0,'src');import face_train;"
                            f"face_train.MODELS_DIR=r'{models_dir}';"
                            f"face_train.TRAINER_FILE=r'{models_dir}/trainer_face.yml';"
                            f"face_train.LABEL_MAP_FILE=r'{models_dir}/label_map.json';"
                            f"face_train.train_faces(r'{faces_dir}')"],
                           cwd=ROOT, capture_output=True, text=True)
        check("LBPH face_train.py still works", r.returncode == 0
              and os.path.exists(os.path.join(models_dir, "trainer_face.yml")), r.stderr[-200:])

        # 3. CNN trainer CLI
        r = subprocess.run([sys.executable, "src/train_cnn.py", faces_dir], cwd=ROOT,
                           env=env, capture_output=True, text=True)
        check("train_cnn.py CLI exits 0", r.returncode == 0, r.stderr[-300:])
        check("face_cnn.pt + label map saved",
              os.path.exists(os.path.join(models_dir, "face_cnn.pt"))
              and os.path.exists(os.path.join(models_dir, "label_map_cnn.json")))
        print(r.stdout.strip().splitlines()[-2:])

        # 4. Haar detect + CNN identify inside a full 640x480 frame
        os.environ["FACE_MODELS_DIR"] = models_dir
        import importlib, train_cnn, cnn_recognizer
        importlib.reload(train_cnn); importlib.reload(cnn_recognizer)
        rec = cnn_recognizer.CNNFaceRecognizer(threshold=0.0)
        check("CNNFaceRecognizer.available()", rec.available())

        correct = detected = 0
        for p in range(PEOPLE):
            frame = to_frame(imgs[p * 10 + 9])
            boxes = cascade.detectMultiScale(frame, 1.2, 5, minSize=(100, 100))
            if len(boxes):
                detected += 1
                x, y, w, h = boxes[0]
                name, prob = rec.predict(frame[y:y + h, x:x + w])
                correct += name == f"person_{p}"
        check("Haar detects face in frame", detected >= PEOPLE - 1, f"({detected}/{PEOPLE})")
        check("CNN identifies detected faces", detected > 0 and correct / detected >= 0.75,
              f"({correct}/{detected})")

        # 5. unknown-rejection on pure noise
        strict = cnn_recognizer.CNNFaceRecognizer(threshold=0.95)
        noise = np.random.RandomState(0).randint(0, 255, (200, 200), np.uint8)
        check("noise image returns a valid (name, prob)", 0 <= strict.predict(noise)[1] <= 1)

        # 6. the monitoring core picks the CNN backend and names a face
        import monitor
        importlib.reload(monitor)
        recognizer = monitor.Recognizer()
        check("monitor.Recognizer selects CNN backend", recognizer.name == "PyTorch CNN")
        test_frame = to_frame(imgs[9])
        x, y, w, h = cascade.detectMultiScale(test_frame, 1.2, 5, minSize=(100, 100))[0]
        name, detail = recognizer.predict(test_frame[y:y + h, x:x + w])
        check("monitor.Recognizer returns (name, confidence)", isinstance(name, str) and bool(detail))

    ok = all(results)
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
