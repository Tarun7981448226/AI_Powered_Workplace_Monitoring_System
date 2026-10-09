# AI Powered Workplace Monitoring System

A computer-vision attendance system with **sign recognition**. It finds faces on
a webcam, identifies people with a **PyTorch convolutional neural network** (or a
classical OpenCV LBPH model), tracks how long each person is present during a
timed session and writes an attendance report. At the same time it reads **hand
signs** and shows what the person is signing in a side panel (and speaks it
aloud on macOS), so people who cannot speak can be understood. Access is gated
behind a teacher login.

## The four files (run them in this order)

| Step | File | What it does |
|---|---|---|
| 1 | `src/face_taker.py` | Enroll a person: captures 30 face photos (look straight, left, right) |
| 2 | `src/face_train.py` | Trains the face CNN + LBPH on your photos, and the sign model on the public HaGRID dataset |
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

## Hand signs (sign -> text)

You do not record any signs. `face_train.py` learns them from the public
**HaGRID** hand-gesture dataset (31,833 images, 18 gestures plus a "no gesture"
class): MediaPipe extracts 21 hand landmarks from ~12,300 images, and a PyTorch
network classifies the hand pose. The extracted landmarks are stored in
`dataset/signs/hagrid_landmarks.npz` (2.6 MB), so training does not need to
download the 1 GB images again. Held-out accuracy on the dataset: **96.7%**.

While the camera runs, a side panel shows the recognised word, a list of words
said so far, and speaks each new word aloud (macOS). Meanings (edit `SIGN_WORDS`
in `src/face_recognizer.py` to change them):

| Show this | Shows |
|---|---|
| Open palm | Hi / Hello |
| Thumbs up / down | Good / Bad |
| Fist | Yes |
| OK sign | OK |
| Stop (palm toward camera) | Stop |
| Peace (V) | Peace |
| Call-me (thumb + pinky) | Call me |
| Mute (finger on lips) | Quiet |
| 1 / 2 / 3 / 4 fingers | One / Two / Three / Four |
| Rock (index + pinky) | Rock on |

These are single hand poses, not full sign language. Real sign languages
(ASL, ISL ...) use motion, two hands and facial expression and need far more data.

Dataset: HaGRID, Kapitanov et al., CC BY-SA 4.0 -
https://github.com/hukenovs/hagrid (sample used: huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p).

## Reading the on-screen text

Each face shows `Name (number)`. For the CNN the number is the confidence in
percent (higher is better, must be at least 70%). For LBPH it is `d=` distance (lower is
better, must be < 70). If you see "Unknown" for yourself, re-enroll in your
current lighting (`face_taker.py`) and train again (`face_train.py`).

## Project structure

```
src/        face_taker.py, face_train.py, face_recognizer.py, main.py
assets/     YuNet face detector, Haar cascade, MediaPipe hand model
dataset/    faces/ (git-ignored, biometric) and signs/ (public landmark data)
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
