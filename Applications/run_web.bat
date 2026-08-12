@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo =====================================================
echo  FALLGUARD AI - DJANGO CAMERA WEB
echo =====================================================
python check_environment.py
if errorlevel 1 goto :error
python manage.py migrate --noinput
if errorlevel 1 goto :error
start "" http://127.0.0.1:8000
python manage.py runserver 127.0.0.1:8000
exit /b 0
:error
echo.
echo [LOI] Chua du dieu kien chay. Hay xem thong bao phia tren.
pause
exit /b 1
