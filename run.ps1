# iRacing AI Paint Generator launcher (PowerShell)
# If you get an execution-policy error, use run.bat instead, or run:
#   powershell -ExecutionPolicy Bypass -File run.ps1

$ports = 7860..7869
foreach ($port in $ports) {
    Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique |
        ForEach-Object {
            Write-Host "Stopping process $_ on port $port"
            Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
        }
}

Set-Location $PSScriptRoot
Write-Host "Opening app in a new window — leave it open while you use the app."
Start-Process cmd -ArgumentList '/k', "cd /d `"$PSScriptRoot`" && python app.py"
Start-Sleep -Seconds 3
Start-Process "http://127.0.0.1:7860"
Write-Host "Browser opened. If it fails to load, wait a few seconds and refresh."