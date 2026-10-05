@echo off
rem 5-fold cross validation: fair comparison with ddddocr, and finds likely label mistakes.
rem Result: models\jpp_cnn_cv_report.txt, plus a read-only model "jpp_cnn_cv" in the labeling tool.
chcp 65001 >nul
cd /d "%~dp0"

rem Use "python" from PATH; fall back to the Windows "py" launcher
set "PY=python"
where python >nul 2>nul || set "PY=py -3"

%PY% -W ignore train.py --cv 5 %*
echo.
pause
