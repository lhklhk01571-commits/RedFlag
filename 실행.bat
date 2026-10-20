@echo off
cd /d "%~dp0"

:: Check Python
python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11+ from python.org
    pause & exit /b 1
)

:: Check .env
if not exist ".env" (
    echo [ERROR] .env file not found. Copy .env.example to .env and fill in API keys.
    pause & exit /b 1
)

:: Auto-install packages if needed
python -c "import flask, langgraph, openai, anthropic" > nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
)

:: Open browser after 3 sec delay (using ping trick - no chcp issue)
start /min "" cmd /c "ping 127.0.0.1 -n 4 > nul & start http://localhost:5000"

echo Server starting... Browser will open automatically.
echo To stop: close this window or press Ctrl+C
echo.

python app.py

echo.
echo Server stopped.
pause
