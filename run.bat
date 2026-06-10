@echo off
title iRacing AI Paint Generator Launcher
cd /d "%~dp0"

REM Free default Gradio ports from leftover python servers.
for /L %%p in (7860,1,7869) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%p " ^| findstr "LISTENING"') do (
        taskkill /F /PID %%a >nul 2>&1
    )
)

echo Opening iRacing AI Paint Generator in a new window...
echo Leave that window open while you use the app.
start "iRacing AI Paint Generator" cmd /k "cd /d %~dp0 && python app.py"
timeout /t 3 >nul
start http://127.0.0.1:7860
echo Done. If the browser shows an error, wait a few seconds and refresh.
pause