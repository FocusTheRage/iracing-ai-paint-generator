@echo off
echo Stopping iRacing AI Paint Generator servers on ports 7860-7869...
for /L %%p in (7860,1,7869) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%p " ^| findstr "LISTENING"') do (
        echo Stopping PID %%a on port %%p
        taskkill /F /PID %%a >nul 2>&1
    )
)
echo Done.
pause