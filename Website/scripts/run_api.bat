@echo off
:: InsureAI — Windows API Run Script
echo Starting InsureAI Backend API...
echo Access: http://localhost:8000
echo Docs:   http://localhost:8000/docs
echo.
cd /d "%~dp0\.."
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
