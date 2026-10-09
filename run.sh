#!/usr/bin/env bash
# Usage: ./run.sh <command>
#   enroll   capture face + gesture data for a new person (webcam)
#   train    train the LBPH model and, when possible, the PyTorch CNN
#   live     live recognition preview with confidence values (ESC to stop)
#   app      login window + timed attendance session
#   test     run all automated tests (no webcam needed)
set -e
cd "$(dirname "$0")"
PY=./venv/bin/python
[ -x "$PY" ] || { echo "Run ./setup.sh first."; exit 1; }

case "$1" in
  enroll) $PY src/face_taker.py ;;
  train)
    $PY src/face_train.py
    n=$(find dataset/faces -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
    if [ "$n" -lt 2 ]; then
      echo "[INFO] Only $n person enrolled - adding public background faces so the CNN can train."
      $PY src/add_background_faces.py
    fi
    $PY src/train_cnn.py ;;
  live)  $PY src/face_recognizer.py ;;
  app)   $PY src/main.py ;;
  test)  $PY tests/test_units.py && $PY tests/test_gui.py && $PY tests/test_pipeline.py && $PY tests/evaluate_cnn.py ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
