@echo off
setlocal EnableDelayedExpansion

where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not in PATH. Please install Python first.
    echo Download from https://www.python.org/downloads/
    pause
    exit /b 1
)

where pip >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: pip is not installed. Ensure pip is available with your Python installation.
    pause
    exit /b 1
)

python -m pip install --upgrade pip
pip install websockets torch transformers numpy tk
pip install torch --index-url https://download.pytorch.org/whl/cu118

where choco >nul 2>&1
if %ERRORLEVEL% equ 0 (
    choco install ffmpeg -y
) else (
    echo Chocolatey not found. Please install FFmpeg manually from https://ffmpeg.org/download.html and add to PATH.
)

echo.
echo IMPORTANT: Additional manual steps required:
echo 1. Install Ollama from https://ollama.ai/ and run:
echo    ollama pull llava:7b
echo    ollama pull llama3.2
echo    ollama serve
echo 2. Install eSpeak for TTS (optional):
echo    Download from http://espeak.sourceforge.net/ or use: choco install espeak

echo Installation complete! Check above for any warnings or manual steps.
pause