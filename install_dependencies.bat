@echo off
setlocal enabledelayedexpansion

:: Create a log file
set LOGFILE=install_log.txt
echo Installation log > %LOGFILE%
echo ================== >> %LOGFILE%

:: Function to check the last command's success
:check_error
if %errorlevel% neq 0 (
    echo Error occurred during %1. Check %LOGFILE% for details.
    echo Error during %1 >> %LOGFILE%
    pause
    exit /b 1
)
goto :eof

:: Check if pip is installed
echo Checking if pip is installed...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Pip is not installed. Please install Python and ensure pip is added to your PATH.
    echo Pip is not installed. >> %LOGFILE%
    pause
    exit /b 1
)
echo Pip is installed. >> %LOGFILE%

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv >> %LOGFILE% 2>&1
call :check_error "creating virtual environment"

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate >> %LOGFILE% 2>&1
call :check_error "activating virtual environment"

:: Install dependencies
echo Installing required packages...

echo Installing tqdm...
pip install tqdm >> %LOGFILE% 2>&1
call :check_error "installing tqdm"

echo Installing opencv-python...
pip install opencv-python >> %LOGFILE% 2>&1
call :check_error "installing opencv-python"

echo Installing scikit-learn...
pip install scikit-learn >> %LOGFILE% 2>&1
call :check_error "installing scikit-learn"

echo Installation complete!
echo Installation complete! >> %LOGFILE%
pause
