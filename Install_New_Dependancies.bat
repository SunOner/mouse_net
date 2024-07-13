@echo off

:: Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo Pip is not installed. Please install Python and ensure pip is added to your PATH.
    pause
    exit /b 1
)

:: Install dependencies
echo Installing required packages...
pip install tqdm
pip install opencv-python
pip install scikit-learn

echo Installation complete!
pause