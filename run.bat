@echo off
setlocal
set "ROOT=%~dp0"

if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT%.venv\Scripts\python.exe"
) else (
    where py >nul 2>nul
    if errorlevel 1 (
        echo Python was not found. Install Python and the packages in requirements.txt.
        exit /b 1
    )
    set "PYTHON=py"
)

"%PYTHON%" "%ROOT%run_detection.py" %*
exit /b %ERRORLEVEL%
