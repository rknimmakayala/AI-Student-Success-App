@echo off
REM Quick script to run both FastAPI backend and Streamlit frontend on Windows

echo 🚀 Starting Enrollment Chatbot Demo
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✓ Python found

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing Streamlit dependencies...
    pip install -r requirements-streamlit.txt
)

REM Start FastAPI backend in background
echo.
echo 1️⃣ Starting FastAPI backend...
start /B cmd /c "cd enrollment_chatbot && python main.py"

REM Wait for API to start
echo    Waiting for API to initialize...
timeout /t 5 /nobreak >nul

REM Check if API is running
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo    ⚠️ API might not be running. Check for errors.
) else (
    echo    ✅ API is running on http://localhost:8000
)

REM Start Streamlit
echo.
echo 2️⃣ Starting Streamlit app...
echo.
streamlit run streamlit_app.py

REM Cleanup happens automatically when window is closed
echo.
echo ✓ Done!
pause
