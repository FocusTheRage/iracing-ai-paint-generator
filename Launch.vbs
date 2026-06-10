' Detached launcher — opens the app in its own window (survives parent process exit).
Set shell = CreateObject("WScript.Shell")
appDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

' Free ports 7860-7869
For port = 7860 To 7869
    cmd = "cmd /c for /f ""tokens=5"" %a in ('netstat -ano ^| findstr "":" & port & " "" ^| findstr LISTENING') do taskkill /F /PID %a >nul 2>&1"
    shell.Run cmd, 0, True
Next

' Start server in a new console window (detached from this launcher).
shell.Run "cmd /k cd /d """ & appDir & """ && python app.py", 1, False

WScript.Sleep 3000
shell.Run "http://127.0.0.1:7860", 1, False

MsgBox "iRacing AI Paint Generator is starting." & vbCrLf & vbCrLf & _
       "A separate console window must stay open while you use the app." & vbCrLf & _
       "Open: http://127.0.0.1:7860" & vbCrLf & vbCrLf & _
       "To stop the server, close that console window or run stop.bat.", _
       vbInformation, "iRacing AI Paint Generator"