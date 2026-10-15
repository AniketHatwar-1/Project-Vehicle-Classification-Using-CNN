@echo off
setlocal
cd /d "%~dp0"
title Vehicle CNN - Improved Training
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" train.py --epochs 30 --batch-size 16 --patience 7
if errorlevel 1 echo Training failed. Check the error above.
echo.
echo Results are saved in the results folder.
pause
