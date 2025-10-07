@echo off
title VRAMancer - Menu Principal Unifié

cls
echo.
echo ████████████████████████████████████████████████████████████████
echo                    VRAMANCER - MENU PRINCIPAL
echo ████████████████████████████████████████████████████████████████
echo.
echo                🚀 Système Unifié de Tests Complets
echo.
echo ════════════════════════════════════════════════════════════════

:menu
cls
echo.
echo ██ VRAMANCER - MENU PRINCIPAL ██
echo.
echo 📊 API Status: 
python -c "
import requests
try:
    r = requests.get('http://localhost:5030/health', timeout=2)
    print('✅ API Active sur port 5030' if r.status_code == 200 else '❌ API Error')
except:
    print('❌ API Non connectée - Lancer en premier!')
"
echo.
echo ════════════════════════════════════════════════════════════════
echo   ÉTAPE 1 - INFRASTRUCTURE
echo ════════════════════════════════════════════════════════════════
echo.
echo 1. 🔧 Lancer API Permanente   (REQUIS EN PREMIER)
echo 2. 🧹 Nettoyer Ports          (Si problèmes)
echo 3. 📋 Status Système          (Diagnostic rapide)
echo.
echo ════════════════════════════════════════════════════════════════
echo   ÉTAPE 2 - INTERFACES PRINCIPALES
echo ════════════════════════════════════════════════════════════════
echo.
echo 10. ⚡ Debug Web Ultra        (Version corrigée)
echo 11. 🖥️  Qt Dashboard           (Interface native)
echo 12. 🌐 Dashboard Web Avancé   (Supervision cluster)
echo 13. 📱 Mobile Dashboard       (Interface mobile)
echo 14. 🔍 Debug Web Simple       (Tests basiques)
echo.
echo ════════════════════════════════════════════════════════════════
echo   ÉTAPE 3 - INTERFACES SPÉCIALISÉES
echo ════════════════════════════════════════════════════════════════
echo.
echo 20. 🪟 System Tray            (Monitoring permanent)
echo 21. 📟 Dashboard CLI          (Ligne de commande)
echo 22. 🎛️  Dashboard Tkinter      (Interface stable)
echo 23. 🚀 Launcher Auto          (Auto-détection)
echo.
echo ════════════════════════════════════════════════════════════════
echo   ÉTAPE 4 - TESTS ET DIAGNOSTIC
echo ════════════════════════════════════════════════════════════════
echo.
echo 30. 🧪 Test Toutes Interfaces (Complet automatique)
echo 31. 🔍 Diagnostic Complet     (Analyse problèmes)
echo 32. 📊 Rapport Final          (Résumé statuts)
echo.
echo ════════════════════════════════════════════════════════════════
echo   MAINTENANCE
echo ════════════════════════════════════════════════════════════════
echo.
echo 90. 🧹 Nettoyer anciens .bat  (Cleanup)
echo 91. 📋 Liste fichiers .bat    (Inventaire)
echo 99. ❌ Quitter
echo.
echo ════════════════════════════════════════════════════════════════

set /p choice="Votre choix (1-99): "

if "%choice%"=="1" goto api_permanente
if "%choice%"=="2" goto nettoyer_ports
if "%choice%"=="3" goto status_systeme
if "%choice%"=="10" goto debug_ultra
if "%choice%"=="11" goto qt_dashboard
if "%choice%"=="12" goto web_avance
if "%choice%"=="13" goto mobile_dashboard
if "%choice%"=="14" goto debug_simple
if "%choice%"=="20" goto system_tray
if "%choice%"=="21" goto cli_dashboard
if "%choice%"=="22" goto tkinter_dashboard
if "%choice%"=="23" goto launcher_auto
if "%choice%"=="30" goto test_toutes
if "%choice%"=="31" goto diagnostic_complet
if "%choice%"=="32" goto rapport_final
if "%choice%"=="90" goto cleanup_bat
if "%choice%"=="91" goto liste_bat
if "%choice%"=="99" goto fin
goto menu

:api_permanente
cls
echo ════════════ LANCEMENT API PERMANENTE ════════════
echo.
echo 🚀 Lancement de l'API permanente...
echo ⚠️  IMPORTANT: Gardez cette fenêtre ouverte
echo.
call api_permanente.bat
pause
goto menu

:debug_ultra
cls
echo ════════════ DEBUG WEB ULTRA ════════════
echo.
echo ⚡ Interface web ultra corrigée
echo 🔧 Boutons fonctionnels garantis
echo.
python debug_web_ultra.py
pause
goto menu

:qt_dashboard
cls
echo ════════════ QT DASHBOARD ════════════
echo.
echo 🖥️ Interface native Qt
echo.
cd /d "%~dp0"
python dashboard\dashboard_qt.py
pause
goto menu

:web_avance
cls
echo ════════════ DASHBOARD WEB AVANCÉ ════════════
echo.
echo 🌐 Supervision cluster avancée
echo 📊 Port 5000 par défaut
echo.
python dashboard\dashboard_web_advanced.py
pause
goto menu

:mobile_dashboard
cls
echo ════════════ MOBILE DASHBOARD ════════════
echo.
echo 📱 Interface optimisée mobile
echo 🌐 Port 5003 par défaut
echo.
python mobile\dashboard_mobile.py
pause
goto menu

:debug_simple
cls
echo ════════════ DEBUG WEB SIMPLE ════════════
echo.
echo 🔍 Interface de test simple
echo.
python debug_web_simple.py
pause
goto menu

:system_tray
cls
echo ════════════ SYSTEM TRAY ════════════
echo.
echo 🪟 Monitoring permanent système
echo.
python dashboard\systray_vramancer.py
pause
goto menu

:cli_dashboard
cls
echo ════════════ CLI DASHBOARD ════════════
echo.
echo 📟 Interface ligne de commande
echo.
set PYTHONPATH=%cd%
python dashboard\dashboard_cli.py
pause
goto menu

:tkinter_dashboard
cls
echo ════════════ TKINTER DASHBOARD ════════════
echo.
echo 🎛️ Interface Tkinter stable
echo.
set PYTHONPATH=%cd%
python dashboard\dashboard_tk.py
pause
goto menu

:launcher_auto
cls
echo ════════════ LAUNCHER AUTO ════════════
echo.
echo 🚀 Auto-détection meilleure interface
echo.
set PYTHONPATH=%cd%
python dashboard\launcher.py
pause
goto menu

:test_toutes
cls
echo ════════════ TEST TOUTES INTERFACES ════════════
echo.
echo 🧪 Test automatique de toutes les interfaces...
echo.
call vramancer_test_complet.bat
pause
goto menu

:diagnostic_complet
cls
echo ════════════ DIAGNOSTIC COMPLET ════════════
echo.
echo 🔍 Analyse complète du système...
echo.
call vramancer_diagnostic.bat
pause
goto menu

:rapport_final
cls
echo ════════════ RAPPORT FINAL ════════════
echo.
echo 📊 Génération rapport de statut...
echo.
call vramancer_rapport.bat
pause
goto menu

:nettoyer_ports
cls
echo ════════════ NETTOYAGE PORTS ════════════
echo.
echo 🧹 Nettoyage ports 5030, 8080, 5000, 5003...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5030" 2^>nul') do taskkill /f /pid %%p >nul 2>&1
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8080" 2^>nul') do taskkill /f /pid %%p >nul 2>&1
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5000" 2^>nul') do taskkill /f /pid %%p >nul 2>&1
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5003" 2^>nul') do taskkill /f /pid %%p >nul 2>&1
echo ✅ Ports nettoyés
pause
goto menu

:status_systeme
cls
echo ════════════ STATUS SYSTÈME ════════════
echo.
python -c "
import sys
print(f'Python: {sys.version.split()[0]}')
try:
    import requests
    print('✅ requests: OK')
except: print('❌ requests: Missing')
try:
    import flask
    print('✅ flask: OK')
except: print('❌ flask: Missing')
try:
    from PyQt5 import QtWidgets
    print('✅ PyQt5: OK')
except: 
    try:
        from PyQt6 import QtWidgets
        print('✅ PyQt6: OK')
    except: print('❌ PyQt: Missing')
"
echo.
echo Ports utilisés:
netstat -an | findstr ":5030 :8080 :5000 :5003"
pause
goto menu

:cleanup_bat
cls
echo ════════════ NETTOYAGE ANCIENS .BAT ════════════
echo.
echo 🧹 Suppression des anciens fichiers .bat...
echo ⚠️  Suppression des fichiers obsolètes uniquement
echo.
del /q test_interfaces.bat 2>nul
del /q test_final.bat 2>nul
del /q lanceur_diagnostic.bat 2>nul
del /q diagnostic_simple.bat 2>nul
del /q test_debug_web_full.bat 2>nul
del /q debug_total.bat 2>nul
del /q simple.bat 2>nul
del /q solution_complete.bat 2>nul
del /q test_qt_dashboard.bat 2>nul
del /q test_debug_web.bat 2>nul
del /q debug_api.bat 2>nul
del /q test_qt_dashboard_full.bat 2>nul
echo ✅ Anciens fichiers .bat supprimés
pause
goto menu

:liste_bat
cls
echo ════════════ LISTE FICHIERS .BAT ════════════
echo.
dir *.bat /b
pause
goto menu

:fin
echo.
echo 👋 Au revoir !
exit
