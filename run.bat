@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run the following commands first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install depthai numpy opencv-python pyside6 mediapipe
    pause
    exit /b 1
)

.venv\Scripts\python.exe -m DaO.main
pause
