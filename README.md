# AI Powered Workplace Monitoring System

A computer-vision attendance and presence-monitoring system. It finds people
on a webcam, identifies them with either a **PyTorch convolutional neural
network** or a classical OpenCV LBPH model, tracks how long each person is
present during a timed session, and writes an attendance report. Access is
gated behind a teacher/admin login.

## How it works

```
webcam frame -> YuNet face detector (neural net, Haar fallback)
             -> crop + normalise
             -> PyTorch CNN (or OpenCV LBPH if no CNN is trained)
             -> per-face smoothing over recent frames
             -> presence timers -> attendance report (Present if >= 80% of session)
```

| Part | File | What it does |
|---|---|---|
| Detection | `src/detector.py` | YuNet (pretrained ONNX CNN, `assets/`) with Haar cascade fallback |
| CNN model | `src/cnn_model.py` | `FaceCNN`: 4 conv blocks (BatchNorm + ReLU), 128-d embedding, classifier |
| CNN training | `src/train_cnn.py` | Augmentation, stratified train/val split, AdamW + OneCycle, saves `models/face_cnn.pt` |
| CNN inference | `src/cnn_recognizer.py` | Softmax confidence, threshold 0.7, "Unknown" below it |
| Core logic | `src/monitor.py` | Backend choice, face tracking/smoothing, presence timers, report |
| Login | `src/auth.py`, `src/main.py` | Salted PBKDF2 password hashes, Tkinter UI |
| Enrollment | `src/face_taker.py` | Guided face capture (+ MediaPipe hand-gesture capture) |
| LBPH baseline | `src/face_train.py` | OpenCV LBPH recognizer |

If `models/face_cnn.pt` exists the CNN is used; otherwise the app falls back to
LBPH automatically.

## Setup (once)

```bash
./setup.sh
```

This picks a Python that includes Tkinter (python.org 3.12/3.13 — Homebrew's
3.14 does not), creates `./venv`, installs `requirements.txt` and runs the unit
tests. On macOS, allow camera access for your terminal / VS Code when asked.

## Use

```bash
./run.sh enroll   # 1. capture faces for each person (webcam)
./run.sh train    # 2. train LBPH + PyTorch CNN
./run.sh live     # 3. optional: live preview with confidence values (ESC stops)
./run.sh app      # 4. login window -> timed attendance session
./run.sh test     # run every automated test (no webcam needed)
```

- First launch of the app asks you to create the teacher account
  (email + password of at least 6 characters). Passwords are stored as salted
  PBKDF2-SHA256 hashes; accounts from older versions are upgraded on first login.
- Set the session length (HH:MM:SS), log in, and monitoring starts. Press ESC
  to stop early. The report is written to `data/attendance_report.json`
  (`Present` if seen for at least 80% of the session; enrolled people who never
  appeared are listed as `Absent`).
- On screen, each face shows `name (confidence)`: the CNN probability (higher
  is better) or the LBPH distance `d=` (lower is better, cutoff 70).

### CNN needs at least two classes

A classifier trained on one person would "recognise" everyone as that person.
`./run.sh train` therefore adds public background faces
(`src/add_background_faces.py`, Olivetti set, stored as `dataset/faces/_background_N`)
when fewer than two people are enrolled. They are reported as "Unknown" and are
ignored by LBPH. Enroll real people to replace them.

## Testing

| Command | What it checks |
|---|---|
| `python tests/test_units.py` | Password hashing/login, report maths, face tracking, detector |
| `python tests/test_pipeline.py` | Dataset -> LBPH + CNN training -> detection -> recognition -> CNN backend selection |
| `python tests/evaluate_cnn.py` | CNN vs LBPH accuracy on the public Olivetti faces (held-out split) |
| `python tests/live_webcam_test.py Name 12` | Live webcam: how often the enrolled person is recognised |

Reference result on Olivetti (40 people, 60/40 split): CNN 98.8%, LBPH 97.5%.
That set is small and clean, so treat the gap as roughly equal; results on your
own data depend on lighting, camera and number of samples per person.

## Project structure

```
src/        application code (see table above)
assets/     YuNet ONNX detector + Haar cascade
tests/      unit, pipeline, benchmark and live-webcam tests
dataset/    captured faces/hands            (git-ignored, biometric data)
models/     trained LBPH + CNN models       (git-ignored)
data/       teacher.db, attendance report   (git-ignored)
```

## Tech stack

[PyTorch](https://pytorch.org/) / torchvision · [OpenCV](https://opencv.org/)
(YuNet, LBPH) · [MediaPipe](https://developers.google.com/mediapipe) ·
NumPy / Pillow · pyttsx3 (offline speech) · SQLite · Tkinter

## Notes & limitations

- Recognition accuracy depends on lighting, camera quality and the number and
  variety of enrolled samples. Enroll under the lighting you will monitor in.
- Re-enroll and retrain if you change the detector or camera setup; the
  enrolled crops should come from the same detector used at run time.
- This is a learning/demo project, not production-grade identity verification.
- All biometric and personal data stays local; nothing is uploaded anywhere.

## License

This project is licensed under the [MIT License](LICENSE).
