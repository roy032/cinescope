@echo off
title CineScope: Kaggle training jobs
cd /d "%~dp0"
if not exist "%USERPROFILE%\.kaggle\kaggle.json" if exist "%USERPROFILE%\Downloads\kaggle.json" (
  mkdir "%USERPROFILE%\.kaggle" 2>nul
  move "%USERPROFILE%\Downloads\kaggle.json" "%USERPROFILE%\.kaggle\kaggle.json" >nul
)
if not exist "%USERPROFILE%\.kaggle\kaggle.json" (
  echo Kaggle API token not found.
  echo kaggle.com - Settings - API - Create New Token. It downloads kaggle.json; then run this file again.
  pause
  exit /b 1
)
.venv\Scripts\python -m pip install -q kaggle
.venv\Scripts\python kaggle\run_all.py
echo.
echo Finished. Scroll up for the summary.
pause
