@echo off
setlocal
cd /d "%~dp0slides"

echo ========================================================================
echo   BOLT SLIDES -- VLM-IDP RESEARCH PRESENTATION DECK
echo   Interactive Presentation Engine (Vite + React)
echo ========================================================================
echo.
echo [INFO] Starting Bolt Slides dev server...
echo [INFO] Presentation will open automatically at http://localhost:5173
echo.

start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:5173"
npm run dev
