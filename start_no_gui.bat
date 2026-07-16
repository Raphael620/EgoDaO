@echo off
cd /d "%~dp0"
start "" /min "%~dp0dist\main.dist\main.exe" --no-gui
