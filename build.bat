@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "MEDIAPIPE_DLL=%VENV_DIR%\Lib\site-packages\mediapipe\tasks\c\libmediapipe.dll"

echo === Cleaning old dist ===
if exist "%PROJECT_DIR%dist\EgoDaO.build" rmdir /s /q "%PROJECT_DIR%dist\EgoDaO.build"
if exist "%PROJECT_DIR%dist\EgoDaO.dist" rmdir /s /q "%PROJECT_DIR%dist\EgoDaO.dist"

echo === Building with Nuitka ===
"%VENV_PYTHON%" -m nuitka ^
  --standalone ^
  --enable-plugin=pyside6 ^
  --windows-icon-from-ico="%PROJECT_DIR%icon.ico" ^
  --include-data-dir="%PROJECT_DIR%DaO\=DaO\" ^
  --include-data-files="%MEDIAPIPE_DLL%=mediapipe/tasks/c/libmediapipe.dll" ^
  --include-package=DaO ^
  --assume-yes-for-downloads ^
  --nofollow-import-to=torch ^
  --output-filename=EgoDaO ^
  --output-dir="%PROJECT_DIR%dist" ^
  "%PROJECT_DIR%DaO\main.py"

if %ERRORLEVEL% NEQ 0 (
    echo === BUILD FAILED ===
    exit /b 1
)

echo.
echo === Copying config.json ===
if exist "%PROJECT_DIR%config.json" copy /y "%PROJECT_DIR%config.json" "%PROJECT_DIR%dist\EgoDaO.dist\config.json" >nul

echo.
echo === Build complete ===
echo EXE: %PROJECT_DIR%dist\EgoDaO.dist\EgoDaO.exe
