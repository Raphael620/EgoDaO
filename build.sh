#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

# MediaPipe task library — Linux uses .so, not .dll
MEDIAPIPE_LIB_DIR="$VENV_DIR/lib/python3.12/site-packages/mediapipe/tasks/c"
# try to find the exact .so file (name may vary)
MEDIAPIPE_SO=$(ls "$MEDIAPIPE_LIB_DIR"/libmediapipe*.so 2>/dev/null | head -1 || echo "")

echo "=== Cleaning old dist ==="
rm -rf "$PROJECT_DIR/dist/EgoDaO.build" "$PROJECT_DIR/dist/EgoDaO.dist" \
       "$PROJECT_DIR/dist/main.build" "$PROJECT_DIR/dist/main.dist"

echo "=== Building with Nuitka ==="
NUIKKA_ARGS=(
  --standalone
  --enable-plugin=pyside6
  --output-filename=EgoDaO
  --include-data-dir="$PROJECT_DIR/DaO/=DaO/"
  --include-package=DaO
  --assume-yes-for-downloads
  --nofollow-import-to=torch
  --output-dir="$PROJECT_DIR/dist"
)

# Include MediaPipe .so if found
if [ -n "$MEDIAPIPE_SO" ]; then
  NUIKKA_ARGS+=(--include-data-files="$MEDIAPIPE_SO=mediapipe/tasks/c/$(basename "$MEDIAPIPE_SO")")
else
  echo "WARNING: libmediapipe .so not found at $MEDIAPIPE_LIB_DIR — MediaPipe may not work in compiled binary"
fi

"$VENV_PYTHON" -m nuitka "${NUIKKA_ARGS[@]}" "$PROJECT_DIR/DaO/main.py"

if [ -d "$PROJECT_DIR/dist/main.build" ]; then
  mv "$PROJECT_DIR/dist/main.build" "$PROJECT_DIR/dist/EgoDaO.build"
fi
if [ -d "$PROJECT_DIR/dist/main.dist" ]; then
  mv "$PROJECT_DIR/dist/main.dist" "$PROJECT_DIR/dist/EgoDaO.dist"
fi

echo
echo "=== Copying config.json ==="
if [ -f "$PROJECT_DIR/config.json" ]; then
  cp "$PROJECT_DIR/config.json" "$PROJECT_DIR/dist/EgoDaO.dist/config.json"
fi

echo
echo "=== Build complete ==="
echo "Binary: $PROJECT_DIR/dist/EgoDaO.dist/EgoDaO.bin"
echo
echo "Run directly:   ./dist/EgoDaO.dist/EgoDaO.bin"
echo "Run headless:   ./dist/EgoDaO.dist/EgoDaO.bin --no-gui"
