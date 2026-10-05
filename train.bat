@echo off
rem Train the captcha model. Result goes to models\jpp_cnn.onnx (shows up in the labeling tool).
chcp 65001 >nul
cd /d "%~dp0"

rem Use "python" from PATH; fall back to the Windows "py" launcher
set "PY=python"
where python >nul 2>nul || set "PY=py -3"

%PY% -W ignore train.py %*
echo.
pause
