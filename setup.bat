@echo off
setlocal
cd /d "%~dp0"
title Vehicle CNN - Setup
py -3.13 --version >nul 2>&1
if errorlevel 1 (
  py -3.14 --version >nul 2>&1
  if errorlevel 1 (echo Python 3.13 or 3.14 not found. Check py -0p.& pause& exit /b 1)
  set PYVER=3.14
) else (
  set PYVER=3.13
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating environment with Python %PYVER%...
  py -%PYVER% -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Next: run train.bat
pause
exit /b 0
:failed
echo Setup failed. Check the error above.
pause
exit /b 1
