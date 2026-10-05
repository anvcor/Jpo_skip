@echo off
rem J-PlatPat captcha labeling - local web UI
rem Usage: double-click. Optional: start_label.bat --dir D:\path\to\jpp_captcha
chcp 65001 >nul
cd /d "%~dp0"

rem Use "python" from PATH; fall back to the Windows "py" launcher
set "PY=python"
where python >nul 2>nul || set "PY=py -3"

%PY% label_server.py %*
if errorlevel 1 pause
