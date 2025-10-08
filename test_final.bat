@echo off
echo ========================================
echo    TEST VALIDATION VRAMANCER FINAL
echo ========================================
echo.

echo [1/5] Test API...
timeout /t 2 >nul
curl -s http://localhost:5030/health > nul 2>&1
if %errorlevel%==0 (
    echo ✅ API fonctionnelle sur port 5030
) else (
    echo ❌ API non accessible - lancez api_permanente.bat
)

echo.
echo [2/5] Test Dashboard Web Avancé...
timeout /t 1 >nul
curl -s http://localhost:5000 > nul 2>&1
if %errorlevel%==0 (
    echo ✅ Dashboard Web Avancé accessible
) else (
    echo ℹ️  Dashboard Web Avancé arrêté
)

echo.
echo [3/5] Test Dashboard Mobile...
timeout /t 1 >nul
curl -s http://localhost:5003 > nul 2>&1
if %errorlevel%==0 (
    echo ✅ Dashboard Mobile accessible
) else (
    echo ℹ️  Dashboard Mobile arrêté
)

echo.
echo [4/5] Test fichiers critiques...
if exist "vramancer.png" (
    echo ✅ Icône system tray présente
) else (
    echo ❌ vramancer.png manquant
)

if exist "systray_vramancer.py" (
    echo ✅ System tray disponible
) else (
    echo ❌ systray_vramancer.py manquant
)

echo.
echo [5/5] Test structure core...
if exist "core\utils.py" (
    echo ✅ Module core.utils présent
) else (
    echo ❌ core\utils.py manquant
)

echo.
echo ========================================
echo        RÉSUMÉ VALIDATION
echo ========================================
echo 🎯 Pour lancer VRAMancer :
echo    1. api_permanente.bat
echo    2. dashboard_web_avance.bat OU systray_vramancer.bat
echo.
echo 🔍 Sur PC Windows pour CUDA :
echo    1. Drivers NVIDIA récents
echo    2. CUDA Toolkit 12.x
echo    3. pip install torch --index-url https://download.pytorch.org/whl/cu121
echo.
pause