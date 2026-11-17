@echo off
title Smart Cafe - Lancement Global
color 0B

echo ======================================================================
echo    SMART CAFE - LANCEMENT COMPLET (FLASK + ANGULAR)
echo ======================================================================
echo.

set PROJ_DIR=C:\Users\rezgu\Desktop\Home\Cours\ING\S3\Projet\Projet _ia
set PY_BIN=C:\Users\rezgu\AppData\Local\Programs\Python\Python311\python.exe

echo [1/2] Demarrage du Backend Flask (Camera 0 ou Video)...
start "Smart Cafe Backend" cmd /k "cd /d %PROJ_DIR% && %PY_BIN% backend\server.py --source 0"

echo En attente du backend (3 secondes)...
ping -n 4 127.0.0.1 >nul

echo [2/2] Demarrage du Frontend Angular (Port 4201)...
start "Smart Cafe Frontend" cmd /k "cd /d %PROJ_DIR%\frontend\smart-cafe-app && npm start"

echo.
echo ======================================================================
echo  Les 2 serveurs sont lances !
echo  - Backend API : http://localhost:5000
echo  - Frontend Web: http://localhost:4201
echo ======================================================================
echo.
pause
