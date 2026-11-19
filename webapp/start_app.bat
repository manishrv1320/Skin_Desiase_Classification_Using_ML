@echo off
rem Starts the web app and opens it in the browser once the model has loaded.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo .venv not found. See README.md, section 1.
  pause
  exit /b 1
)
set TF_CPP_MIN_LOG_LEVEL=2
start "" cmd /c "timeout /t 10 /nobreak >nul & start http://127.0.0.1:5000"
".venv\Scripts\python.exe" app.py
pause
