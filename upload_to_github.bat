@echo off
echo =======================================================
echo   Pushing FaceScope AI Application to GitHub
echo =======================================================
echo.
echo Make sure you have created an empty repository named
echo 'face-detection-app' at: https://github.com/new
echo.
echo Pushing branch 'main' to origin...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =======================================================
    echo [SUCCESS] Code uploaded successfully to GitHub!
    echo View repo at: https://github.com/pranavbpeenikkal/face-detection-app
    echo =======================================================
) else (
    echo.
    echo [ERROR] Push failed. 
    echo Please ensure the repo 'face-detection-app' exists on GitHub:
    echo 1. Open https://github.com/new
    echo 2. Repository Name: face-detection-app
    echo 3. Keep it Public and do NOT initialize with README/license
    echo 4. Click 'Create repository' and run this script again.
)
pause
