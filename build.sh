#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
MEDIAPIPE_DLL="$VENV_DIR/lib/python3.12/site-packages/mediapipe/tasks/c/libmediapipe.dll"

echo "=== Cleaning old dist ==="
rm -rf "$PROJECT_DIR/dist/main.build" "$PROJECT_DIR/dist/main.dist"

echo "=== Building with Nuitka ==="
"$VENV_PYTHON" -m nuitka \
  --standalone \
  --enable-plugin=pyside6 \
  --windows-icon-from-ico="$PROJECT_DIR/icon.ico" \
  --include-data-dir="$PROJECT_DIR/DaO/=DaO/" \
  --include-data-files="$MEDIAPIPE_DLL=mediapipe/tasks/c/libmediapipe.dll" \
  --include-package=DaO \
  --assume-yes-for-downloads \
  --output-dir="$PROJECT_DIR/dist" \
  "$PROJECT_DIR/DaO/main.py"

echo
echo "=== Build complete ==="
echo "EXE: $PROJECT_DIR/dist/main.dist/main.exe"
