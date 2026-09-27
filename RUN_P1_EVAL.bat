@echo off
title CineScope: Phase 1 hand-labelled evaluation
cd /d "%~dp0"
.venv\Scripts\python p1_cnn\eval_own.py || goto :fail
.venv\Scripts\python tools\update_results.py
git add p1_cnn\labels p1_cnn\results p1_cnn\README.md README.md
git commit -m "Phase 1: hand-labelled evaluation on held-out episodes"
echo.
echo Done: the tables in p1_cnn\README.md are filled in and committed locally.
pause
exit /b 0
:fail
pause
exit /b 1
