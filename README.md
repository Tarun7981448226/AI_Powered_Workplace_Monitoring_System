# AI Powered Workplace Monitoring System

A computer-vision attendance system. It finds faces on a webcam, identifies
people with a **PyTorch convolutional neural network** (or a classical OpenCV
LBPH model), tracks how long each person is present during a timed session and
writes an attendance report. Access is gated behind a teacher login.

## The four files (run them in this order)

| Step | File | What it does |
|---|---|---|
| 1 | `src/face_taker.py` | Enroll a person: captures 30 face photos (look straight, left, right) and hand-gesture images |
| 2 | `src/face_train.py` | Trains the PyTorch CNN and the OpenCV LBPH model on the captured photos |
| 3 | `src/face_recognizer.py` | Live preview: shows `name (confidence)` on the camera feed, ESC to stop |
| 4 | `src/main.py` | The full system: teacher login, then a timed attendance session |

**In VS Code:** open the file and click the ▶ Run button (top right), or run it
from the terminal, for example `python src/face_taker.py`.

## Setup (once)

Use a Python that includes Tkinter (python.org 3.12 / 3.13; Homebrew's 3.14 does not):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

In VS Code choose the `venv` interpreter (bottom-right). On macOS allow camera
access for VS Code / your terminal when asked.

## How it works

```
camera frame -> YuNet face detector (neural net, Haar fallback)
             -> crop + normalise
             -> PyTorch CNN (LBPH if no CNN is trained)
             -> per-face smoothing over recent frames
             -> presence timers -> attendance report
```

- **CNN:** 4 convolution blocks (BatchNorm + ReLU), 128-d embedding, trained
  with augmentation (rotation, shift, brightness, flip), AdamW + OneCycle, saved
  to `models/face_cnn.pt`. A match needs a confidence of at least 70%.
- **LBPH fallback:** used when no CNN is trained. A match needs distance < 70.
- **Only one person enrolled?** A CNN needs at least two classes, so
  `face_train.py` adds a few public background faces (Olivetti set, stored as
  `dataset/faces/_background_N`, needs scikit-learn). They are reported as
  "Unknown" and LBPH ignores them. Enroll more real people to replace them.
- **Report:** a person is **Present** if seen for >= 80% of the session, else
  **Absent**; enrolled people who never appear are listed as Absent. Saved to
  `data/attendance_report.json`.
- **Login:** the first launch of `main.py` asks you to create the teacher
  account (email + password of 6+ characters). Passwords are stored as salted
  PBKDF2 hashes.

## Reading the on-screen text

Each face shows `Name (number)`. For the CNN the number is the confidence in
percent (higher is better, must be at least 70%). For LBPH it is `d=` distance (lower is
better, must be < 70). If you see "Unknown" for yourself, re-enroll in your
current lighting (`face_taker.py`) and train again (`face_train.py`).

## Project structure

```
src/        face_taker.py, face_train.py, face_recognizer.py, main.py
assets/     YuNet ONNX face detector + Haar cascade
dataset/    captured faces and hands   (git-ignored, biometric data)
models/     trained models             (git-ignored)
data/       teacher.db, attendance report (git-ignored)
```

## Notes & limitations

- Accuracy depends on lighting, camera and the variety of enrolled photos.
- Retrain after re-enrolling; enrolled crops should come from the same detector
  used at run time (YuNet).
- A learning/demo project, not production-grade identity verification.
- All biometric data stays local; nothing is uploaded anywhere.

## License

[MIT](LICENSE)
