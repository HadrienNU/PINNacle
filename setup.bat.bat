@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo      PINNacle Setup Script (Windows)
echo ========================================

REM Decide which venv folder to use
set "VENV_DIR=venv"

REM If a Linux/macOS venv exists in ./venv (with bin/), don't try to use it on Windows.
if exist "%VENV_DIR%\bin\activate" (
    echo Detected a Linux/macOS virtual environment at "%VENV_DIR%".
    echo Creating/using a separate Windows virtual environment: "venv_win".
    set "VENV_DIR=venv_win"
)

REM Create venv if needed
if not exist "%VENV_DIR%" (
    echo Creating virtual environment in "%VENV_DIR%"...
    python -m venv "%VENV_DIR%"
)

REM Activate virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

REM Install requirements
echo Installing requirements...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo Setup complete. Virtual environment: "%VENV_DIR%"
endlocal
