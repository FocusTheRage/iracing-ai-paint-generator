@echo off
REM Wrapper for run.ps1 when PowerShell script execution is blocked.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
pause