@echo off
chcp 65001 >nul
title VRAMancer - Lanceur Rapide

cls
echo.
echo ████████████████████████████████████████████████████████████████
echo                   VRAMANCER - LANCEUR RAPIDE
echo ████████████████████████████████████████████████████████████████
echo.
echo          🚀 Acces rapide aux fonctions principales
echo.

:menu
cls
echo.
echo ██ VRAMANCER - LANCEUR RAPIDE ██
echo.

REM Test API rapide
python -c "
import requests
try:
    r = requests.get('http://localhost:5030/health', timeout=1)
    print('📊 API Status: ✅ ACTIVE sur port 5030' if r.status_code == 200 else '📊 API Status: ❌ ERREUR')
except:
    print('📊 API Status: ❌ NON CONNECTEE')
" 2>nul

echo.
echo ════════════════════════════════════════════════════════════════
echo   🔧 INFRASTRUCTURE
echo ════════════════════════════════════════════════════════════════
echo.
echo 1. 🚀 API Permanente           (Requis en premier)
echo 2. 🧪 Test Toutes Interfaces   (Diagnostique tout)
echo 3. 📋 Menu Complet             (Toutes les options)
echo.
echo ════════════════════════════════════════════════════════════════
echo   ⚡ INTERFACES PRINCIPALES
echo ════════════════════════════════════════════════════════════════
echo.
echo 10. 🖥️  Qt Dashboard            (Interface native - RECOMMANDEE)
echo 11. ⚡ Debug Web Ultra         (Interface web corrigee)
echo 12. 🌐 Dashboard Web Avance    (Supervision cluster)
echo 13. 📱 Mobile Dashboard        (Interface mobile)
echo.
echo ════════════════════════════════════════════════════════════════
echo   🛠️ UTILITAIRES
echo ════════════════════════════════════════════════════════════════
echo.
echo 20. 🧹 Nettoyer Ports          (Si problemes)
echo 21. 📋 Status Systeme          (Diagnostic rapide)
echo.
echo 99. ❌ Quitter
echo.
echo ════════════════════════════════════════════════════════════════

set /p choice="Votre choix (1-99): "

if "%choice%"=="1" goto api_permanente
if "%choice%"=="2" goto test_complet
if "%choice%"=="3" goto menu_complet
if "%choice%"=="10" goto qt_dashboard
if "%choice%"=="11" goto debug_ultra
if "%choice%"=="12" goto web_avance
if "%choice%"=="13" goto mobile_dashboard
if "%choice%"=="20" goto nettoyer_ports
if "%choice%"=="21" goto status_systeme
if "%choice%"=="99" goto fin
goto menu

:api_permanente
cls
echo ════════════ LANCEMENT API PERMANENTE ════════════
echo.
echo 🚀 Lancement de l'API permanente...
echo ⚠️  Une nouvelle fenetre va s'ouvrir - GARDEZ-LA OUVERTE
echo ✅ Revenez ici pour utiliser les autres options
echo.
start "VRAMancer API Permanente" cmd /c "cd /d "%~dp0" && api_permanente.bat"
echo.
echo ✅ API lancee dans une nouvelle fenetre
echo 📋 Attendez 5 secondes puis utilisez les autres options
timeout /t 5 /nobreak >nul
goto menu

:test_complet
cls  
echo ════════════ TEST COMPLET STANDALONE ════════════
echo.
echo 🧪 Test automatique de toutes les interfaces
echo 📋 Rapport detaille genere
echo ⚠️  Peut etre lance meme sans API active
echo.
start "Test Complet" cmd /c "cd /d "%~dp0" && vramancer_test_standalone.bat"
echo.
echo ✅ Test lance dans une nouvelle fenetre
timeout /t 2 /nobreak >nul
goto menu

:menu_complet
cls
echo ════════════ MENU COMPLET ════════════
echo.
echo 📋 Ouverture du menu complet avec toutes les options...
echo.
start "Menu Complet" cmd /c "cd /d "%~dp0" && vramancer_menu_principal_fixe.bat"
echo.
echo ✅ Menu complet lance dans une nouvelle fenetre
timeout /t 2 /nobreak >nul
goto menu

:qt_dashboard
cls
echo ════════════ QT DASHBOARD ════════════
echo.
echo 🖥️ Interface native Qt (RECOMMANDEE)
echo ⚠️  Necessite l'API active sur port 5030
echo.
start "Qt Dashboard" cmd /c "cd /d "%~dp0" && python dashboard\dashboard_qt.py && pause"
echo.
echo ✅ Qt Dashboard lance
timeout /t 2 /nobreak >nul
goto menu

:debug_ultra
cls
echo ════════════ DEBUG WEB ULTRA ════════════
echo.
echo ⚡ Interface web ultra corrigee
echo 🌐 Ouverture automatique sur http://localhost:8080
echo.
start "Debug Web Ultra" cmd /c "cd /d "%~dp0" && python debug_web_ultra.py && pause"
echo.
echo ✅ Debug Web Ultra lance
timeout /t 2 /nobreak >nul
goto menu

:web_avance
cls
echo ════════════ DASHBOARD WEB AVANCE ════════════
echo.
echo 🌐 Supervision cluster avancee
echo 📊 Ouverture sur http://localhost:5000
echo.
start "Dashboard Web Avance" cmd /c "cd /d "%~dp0" && python dashboard\dashboard_web_advanced.py && pause"
echo.
echo ✅ Dashboard Web Avance lance
timeout /t 2 /nobreak >nul
goto menu

:mobile_dashboard
cls
echo ════════════ MOBILE DASHBOARD ════════════
echo.
echo 📱 Interface mobile responsive
echo 🌐 Ouverture sur http://localhost:5003
echo.
start "Mobile Dashboard" cmd /c "cd /d "%~dp0" && python mobile\dashboard_mobile.py && pause"
echo.
echo ✅ Mobile Dashboard lance
timeout /t 2 /nobreak >nul
goto menu

:nettoyer_ports
cls
echo ════════════ NETTOYAGE PORTS ════════════
echo.
echo 🧹 Nettoyage des ports VRAMancer...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5030" 2^>nul') do taskkill /f /pid %%p >nul 2>&1
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8080" 2^>nul') do taskkill /f /pid %%p >nul 2>&1
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5000" 2^>nul') do taskkill /f /pid %%p >nul 2>&1
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5003" 2^>nul') do taskkill /f /pid %%p >nul 2>&1
echo ✅ Ports nettoyes
timeout /t 3 /nobreak >nul
goto menu

:status_systeme
cls
echo ════════════ STATUS SYSTEME RAPIDE ════════════
echo.
python -c "
import sys
print(f'Python: {sys.version.split()[0]}')

# Test dependances critiques
deps = ['requests', 'flask']
for dep in deps:
    try:
        __import__(dep)
        print(f'✅ {dep}: OK')
    except: 
        print(f'❌ {dep}: MANQUANT')

# Test Qt
qt_found = False
for qt in ['PyQt5.QtWidgets', 'PyQt6.QtWidgets', 'PySide2.QtWidgets', 'PySide6.QtWidgets']:
    try:
        __import__(qt)
        print(f'✅ Qt: {qt.split(\".\")[0]} OK')
        qt_found = True
        break
    except:
        continue
if not qt_found:
    print('❌ Qt: AUCUNE LIBRAIRIE TROUVEE')

# Test API
try:
    import requests
    r = requests.get('http://localhost:5030/health', timeout=1)
    print(f'✅ API: ACTIVE (status {r.status_code})')
except:
    print('❌ API: NON CONNECTEE')
"
echo.
echo Appuyez sur une touche pour continuer...
pause >nul
goto menu

:fin
cls
echo.
echo 👋 Au revoir !
echo.
echo 📋 RAPPEL - Fichiers principaux:
echo    - vramancer_lanceur_rapide.bat    (ce menu)
echo    - api_permanente.bat              (API - LANCER EN PREMIER)
echo    - vramancer_test_standalone.bat   (test complet)
echo    - vramancer_menu_principal_fixe.bat (menu complet)
echo.
echo 💡 CONSEIL: Lancez toujours l'API permanente en premier !
echo.
pause
exit