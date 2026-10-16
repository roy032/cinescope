@echo off
title CineScope: set up, test and run everything
cd /d "%~dp0"
rem Any arguments are passed on: RUN_EVERYTHING.bat test pipeline --clip trailers\any_clip.mp4
python tools\run_everything.py %*
echo.
echo Finished. Scroll up for the summary.
pause
