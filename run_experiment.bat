@echo off
REM Запуск из корня проекта stereo_middlebury_project
python -m src.main --download
python -m src.main
pause
