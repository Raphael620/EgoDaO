#!/bin/bash
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/python" ]; then
    echo "[ERROR] .venv not found. Run the following commands first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install depthai numpy opencv-python pyside6 mediapipe"
    exit 1
fi

.venv/bin/python -m DaO.main
