@echo off
REM Runs the suite and writes check_report.txt. A double click is enough.
cd /d "%~dp0"
python check.py
echo.
pause
