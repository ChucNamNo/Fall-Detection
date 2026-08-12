@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo =====================================================
echo  CÀI THƯ VIỆN WEB FALL DETECTION
echo =====================================================
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :error
python check_environment.py
pause
exit /b 0
:error
echo.
echo [LOI] Cai thu vien that bai. Kiem tra Python va ket noi mang.
pause
exit /b 1
