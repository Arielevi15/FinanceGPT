@echo off
title LEVI FINANCE
cd /d "%~dp0"
call venv\Scripts\activate
set STREAMLIT_SERVER_PORT=8501
start "" streamlit run app.py --server.headless true
timeout /t 4 /nobreak > nul
start chrome --app=http://localhost:8501
exit
