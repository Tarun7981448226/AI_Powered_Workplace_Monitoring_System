"""Benchmarks the PyTorch CNN against the OpenCV LBPH baseline.

Uses the public Olivetti faces dataset (40 people x 10 images) so no personal
biometric data is needed. Run from the project root:

    python tests/evaluate_cnn.py
"""
import os
import sys
import tempfile
import time

import cv2
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cnn_recognizer import CNNFaceRecognizer
from train_cnn import preprocess_face, save_model, split_per_class, train_model


def main():
    data = fetch_olivetti_faces(shuffle=False)
    images = (data.images * 255).astype(np.uint8)
    labels = data.target
    label_map = {int(i): f"person_{i:02d}" for i in sorted(set(labels.tolist()))}

    # 60 / 40 split, identical for both models
    train_idx, val_idx = split_per_class(labels, val_fraction=0.4, seed=0)
    tr_x = np.array([preprocess_face(images[i]) for i in train_idx])
    va_x = np.array([preprocess_face(images[i]) for i in val_idx])
    tr_y, va_y = labels[train_idx], labels[val_idx]
    print(f"train={len(tr_x)} val={len(va_x)} classes={len(label_map)}")

    # ---- PyTorch CNN
    t0 = time.time()
    model, history = train_model(tr_x, tr_y, va_x, va_y, len(label_map), epochs=60)
    cnn_train_s = time.time() - t0

    # round-trip through save/load + the real inference class
    with tempfile.TemporaryDirectory() as tmp:
        mf, lf = os.path.join(tmp, "m.pt"), os.path.join(tmp, "l.json")
        save_model(model, label_map, mf, lf)
        rec = CNNFaceRecognizer(mf, lf, threshold=0.0)
        preds = [rec.predict(img)[0] for img in va_x]
        cnn_acc = np.mean([p == label_map[int(y)] for p, y in zip(preds, va_y)])

        # unknown-rejection: a high threshold should reject some low-confidence faces
        strict = CNNFaceRecognizer(mf, lf, threshold=0.9)
        rejected = np.mean([strict.predict(img)[0] == "Unknown" for img in va_x])

    # ---- OpenCV LBPH baseline (same split, same settings as face_train.py)
    lbph = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    t0 = time.time()
    lbph.train([cv2.resize(x, (220, 220)) for x in tr_x], tr_y)
    lbph_train_s = time.time() - t0
    lbph_preds = [lbph.predict(cv2.resize(x, (220, 220)))[0] for x in va_x]
    lbph_acc = np.mean(np.array(lbph_preds) == va_y)

    print("\n===== RESULTS (held-out validation) =====")
    print(f"PyTorch CNN : accuracy={cnn_acc:.3f}  train_time={cnn_train_s:.1f}s  "
          f"final_loss={history[-1][0]:.3f}")
    print(f"LBPH (OpenCV): accuracy={lbph_acc:.3f}  train_time={lbph_train_s:.1f}s")
    print(f"CNN at 0.9 confidence threshold rejected {rejected:.1%} of val faces as Unknown")

    assert cnn_acc > 0.5, "CNN failed to learn"


if __name__ == "__main__":
    main()
