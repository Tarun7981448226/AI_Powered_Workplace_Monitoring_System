#!/usr/bin/env bash
# One-time setup: creates ./venv with everything installed and runs the unit tests.
set -e
cd "$(dirname "$0")"

# Tkinter (the login window) is missing from Homebrew's Python 3.14, so prefer
# python.org 3.13 / 3.12 when they exist.
for candidate in \
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    python3.13 python3.12 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import tkinter" 2>/dev/null; then
    PY="$candidate"; break
  fi
done
[ -n "$PY" ] || { echo "No Python with Tkinter found. Install Python 3.13 from python.org."; exit 1; }

echo "Using $($PY --version) at $(command -v $PY)"
"$PY" -m venv venv
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q -r requirements.txt
./venv/bin/python tests/test_units.py
echo
echo "Setup complete. Next: ./run.sh enroll"
