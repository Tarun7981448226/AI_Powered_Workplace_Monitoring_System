"""Adds public 'background' faces (Olivetti set) as extra CNN classes.

A classifier trained on one person would recognise everybody as that person.
These background identities give the CNN other faces to separate you from; they
are reported as "Unknown" at run time and are ignored by the LBPH trainer.
Replace them with real enrolled people whenever you can.

    python src/add_background_faces.py [count]     (default 6 identities)
"""
import os
import sys

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES_DIR = os.path.join(BASE_DIR, "dataset", "faces")


def main(count=6):
    from sklearn.datasets import fetch_olivetti_faces

    images = (fetch_olivetti_faces(shuffle=False).images * 255).astype(np.uint8)
    saved = 0
    for p in range(count):
        folder = os.path.join(FACES_DIR, f"_background_{p + 1}")
        os.makedirs(folder, exist_ok=True)
        for k in range(10):
            face = cv2.equalizeHist(cv2.resize(images[p * 10 + k], (220, 220)))
            cv2.imwrite(os.path.join(folder, f"{k}.jpg"), face)
            saved += 1
    print(f"[INFO] Added {saved} background images ({count} identities) in {FACES_DIR}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
