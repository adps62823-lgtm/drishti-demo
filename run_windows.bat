@echo off
title DRISHTI — AI Vision Assistant
color 0B
echo.
echo  =========================================
echo   DRISHTI — AI Vision Assistant
echo   by Aditya Pratap Singh
echo  =========================================
echo.
echo  Installing / checking dependencies...
pip install -r requirements.txt --quiet
echo.
echo  Starting DRISHTI server...
echo  Open your browser at: http://localhost:5000
echo.
start "" "http://localhost:5000"
python app.py
pause
