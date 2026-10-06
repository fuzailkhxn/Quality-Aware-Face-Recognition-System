@echo off
echo ============================================
echo  Quality-Aware Face Recognition
echo  Starting Application...
echo ============================================
echo.

call .venv\Scripts\activate 2>nul
if errorlevel 1 (
    echo Virtual environment not found. Trying system Python...
)

echo Launching web interface...
echo.
echo The app will open in your browser automatically.
echo If it does not open, go to: http://localhost:8501
echo Press Ctrl+C in this window to stop the server.
echo.

python -m streamlit run app.py --server.headless false
pause
