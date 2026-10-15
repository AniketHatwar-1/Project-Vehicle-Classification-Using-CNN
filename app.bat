@echo off
setlocal
cd /d "%~dp0"
title Vehicle CNN - App
if not exist ".venv\Scripts\python.exe" (echo Please run setup.bat first.& pause & exit /b 1)
if not exist "models\vehicle_cnn.pth" (echo Run train.bat first.& pause & exit /b 1)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
