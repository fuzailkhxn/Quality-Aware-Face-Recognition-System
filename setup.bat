@echo off
echo ============================================
echo  Quality-Aware Face Recognition - Setup
echo ============================================
echo.

echo [1/3] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Could not create virtual environment.
    echo Make sure Python 3.10+ is installed.
    pause
    exit /b 1
)

echo.
echo [2/3] Activating virtual environment...
call .venv\Scripts\activate

echo.
echo [3/3] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Could not install dependencies.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Setup complete!
echo ============================================
echo.
echo Next steps:
echo   1. Add face images to dataset\registered\PersonName\
echo   2. Run: python scripts\build_database.py
echo   3. Run: streamlit run app.py
echo.
echo Or simply double-click run_demo.bat
echo.
pause
