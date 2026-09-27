@echo off
title CineScope: Phase 1 on our own episodes
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo The .venv folder is missing. Create it first: see README.md
  pause
  exit /b 1
)
if not exist "%USERPROFILE%\.kaggle\kaggle.json" if exist "%USERPROFILE%\Downloads\kaggle.json" (
  mkdir "%USERPROFILE%\.kaggle" 2>nul
  move "%USERPROFILE%\Downloads\kaggle.json" "%USERPROFILE%\.kaggle\kaggle.json" >nul
)
.venv\Scripts\python -m pip install -q kaggle
echo === 1/4 quick self-test ===
.venv\Scripts\python -m pytest -q tests\test_own_shots.py
echo === 2/4 find, copy and cut the episodes (the longest step; safe to re-run) ===
.venv\Scripts\python p1_cnn\build_own_shots.py --search D:\ || goto :fail
echo === 3/4 labelling page for the test shots ===
.venv\Scripts\python p1_cnn\label_scales.py || goto :fail
start "" "outputs\label_scales\index.html"
echo The labelling page is open in your browser. Label while Kaggle trains; press Download labels when done.
echo === 4/4 upload the key frames and train on Kaggle (about 1-2 hours) ===
.venv\Scripts\python kaggle\run_all.py --only p1_own || goto :fail
echo.
echo Finished. After labelling, run RUN_P1_EVAL.bat for the final numbers.
pause
exit /b 0
:fail
echo.
echo A step failed: scroll up for the message.
pause
exit /b 1
