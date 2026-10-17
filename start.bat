@echo off
rem Double-click to start Reel Studio, then use the page that opens.
cd /d "%~dp0"
start "" http://127.0.0.1:5050
".venv\Scripts\python.exe" app.py
pause
