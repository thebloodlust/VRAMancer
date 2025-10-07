@echo off
chcp 65001 >nul
title VRAMancer - Menu Principal Unifie

cls
echo.
echo ████████████████████████████████████████████████████████████████
echo                    VRAMANCER - MENU PRINCIPAL
echo ████████████████████████████████████████████████████████████████
echo.
echo                🚀 Systeme Unifie de Tests Complets
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
    print('❌ API Non connectee - Lancer en premier!')
" 2>nul
echo.
echo ════════════════════════════════════════════════════════════════
echo   ETAPE 1 - INFRASTRUCTURE
echo ════════════════════════════════════════════════════════════════
echo.
echo 1. 🔧 Lancer API Permanente   (REQUIS EN PREMIER)
echo 2. 🧹 Nettoyer Ports          (Si problemes)
echo 3. 📋 Status Systeme          (Diagnostic rapide)
echo.
echo ════════════════════════════════════════════════════════════════
echo   ETAPE 2 - INTERFACES PRINCIPALES
echo ════════════════════════════════════════════════════════════════
echo.
echo 10. ⚡ Debug Web Ultra        (Version corrigee)
echo 11. 🖥️  Qt Dashboard           (Interface native)
echo 12. 🌐 Dashboard Web Avance   (Supervision cluster)
echo 13. 📱 Mobile Dashboard       (Interface mobile)
echo 14. 🔍 Debug Web Simple       (Tests basiques)
echo.
echo ════════════════════════════════════════════════════════════════
echo   ETAPE 3 - INTERFACES SPECIALISEES
echo ════════════════════════════════════════════════════════════════
echo.
echo 20. 🪟 System Tray            (Monitoring permanent)
echo 21. 📟 Dashboard CLI          (Ligne de commande)
echo 22. 🎛️  Dashboard Tkinter      (Interface stable)
echo 23. 🚀 Launcher Auto          (Auto-detection)
echo.
echo ════════════════════════════════════════════════════════════════
echo   ETAPE 4 - TESTS ET DIAGNOSTIC
echo ════════════════════════════════════════════════════════════════
echo.
echo 30. 🧪 Test Toutes Interfaces (Complet automatique)
echo 31. 🔍 Diagnostic Complet     (Analyse problemes)
echo 32. 📊 Rapport Final          (Resume statuts)
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
echo ⚠️  IMPORTANT: Une nouvelle fenetre va s'ouvrir
echo ⚠️  Gardez cette nouvelle fenetre ouverte
echo ⚠️  Revenez ici pour utiliser les autres options
echo.
echo Appuyez sur une touche pour continuer...
pause >nul
start "VRAMancer API Permanente" cmd /c "cd /d "%~dp0" && api_permanente.bat"
echo.
echo ✅ API lancee dans une nouvelle fenetre
echo 📋 Vous pouvez maintenant utiliser les autres options
echo.
pause
goto menu

:debug_ultra
cls
echo ════════════ DEBUG WEB ULTRA ════════════
echo.
echo ⚡ Interface web ultra corrigee
echo 🔧 Boutons fonctionnels garantis
echo 🌐 Ouverture sur http://localhost:8080
echo.
start "Debug Web Ultra" cmd /c "cd /d "%~dp0" && python debug_web_ultra.py && pause"
echo ✅ Debug Web Ultra lance dans une nouvelle fenetre
pause
goto menu

:qt_dashboard
cls
echo ════════════ QT DASHBOARD ════════════
echo.
echo 🖥️ Interface native Qt
echo.
start "Qt Dashboard" cmd /c "cd /d "%~dp0" && python dashboard\dashboard_qt.py && pause"
echo ✅ Qt Dashboard lance dans une nouvelle fenetre
pause
goto menu

:web_avance
cls
echo ════════════ DASHBOARD WEB AVANCE ════════════
echo.
echo 🌐 Supervision cluster avancee
echo 📊 Ouverture sur http://localhost:5000
echo.
start "Dashboard Web Avance" cmd /c "cd /d "%~dp0" && python dashboard\dashboard_web_advanced.py && pause"
echo ✅ Dashboard Web Avance lance dans une nouvelle fenetre
pause
goto menu

:mobile_dashboard
cls
echo ════════════ MOBILE DASHBOARD ════════════
echo.
echo 📱 Interface optimisee mobile
echo 🌐 Ouverture sur http://localhost:5003
echo.
start "Mobile Dashboard" cmd /c "cd /d "%~dp0" && python mobile\dashboard_mobile.py && pause"
echo ✅ Mobile Dashboard lance dans une nouvelle fenetre
pause
goto menu

:debug_simple
cls
echo ════════════ DEBUG WEB SIMPLE ════════════
echo.
echo 🔍 Interface de test simple
echo.
start "Debug Web Simple" cmd /c "cd /d "%~dp0" && python debug_web_simple.py && pause"
echo ✅ Debug Web Simple lance dans une nouvelle fenetre
pause
goto menu

:system_tray
cls
echo ════════════ SYSTEM TRAY ════════════
echo.
echo 🪟 Monitoring permanent systeme
echo.
start "System Tray" cmd /c "cd /d "%~dp0" && python dashboard\systray_vramancer.py && pause"
echo ✅ System Tray lance dans une nouvelle fenetre
pause
goto menu

:cli_dashboard
cls
echo ════════════ CLI DASHBOARD ════════════
echo.
echo 📟 Interface ligne de commande
echo.
start "CLI Dashboard" cmd /c "cd /d "%~dp0" && set PYTHONPATH=%cd% && python dashboard\dashboard_cli.py && pause"
echo ✅ CLI Dashboard lance dans une nouvelle fenetre
pause
goto menu

:tkinter_dashboard
cls
echo ════════════ TKINTER DASHBOARD ════════════
echo.
echo 🎛️ Interface Tkinter stable
echo.
start "Tkinter Dashboard" cmd /c "cd /d "%~dp0" && set PYTHONPATH=%cd% && python dashboard\dashboard_tk.py && pause"
echo ✅ Tkinter Dashboard lance dans une nouvelle fenetre
pause
goto menu

:launcher_auto
cls
echo ════════════ LAUNCHER AUTO ════════════
echo.
echo 🚀 Auto-detection meilleure interface
echo.
start "Launcher Auto" cmd /c "cd /d "%~dp0" && set PYTHONPATH=%cd% && python dashboard\launcher.py && pause"
echo ✅ Launcher Auto lance dans une nouvelle fenetre
pause
goto menu

:test_toutes
cls
echo ════════════ TEST TOUTES INTERFACES ════════════
echo.
echo 🧪 Test automatique de toutes les interfaces...
echo 📋 Rapport detaille genere
echo ⏱️  Duree estimee: 2-3 minutes
echo.
echo Appuyez sur une touche pour demarrer...
pause >nul
start "Test Complet" cmd /c "cd /d "%~dp0" && vramancer_test_complet.bat"
echo.
echo ✅ Test complet lance dans une nouvelle fenetre
echo 📋 Les resultats seront sauves dans vramancer_test_results.txt
echo.
pause
goto menu

:diagnostic_complet
cls
echo ════════════ DIAGNOSTIC COMPLET ════════════
echo.
echo 🔍 Analyse complete du systeme...
echo 📋 Diagnostic detaille genere
echo ⏱️  Duree estimee: 1-2 minutes
echo.
echo Appuyez sur une touche pour demarrer...
pause >nul
start "Diagnostic Complet" cmd /c "cd /d "%~dp0" && vramancer_diagnostic.bat"
echo.
echo ✅ Diagnostic lance dans une nouvelle fenetre
echo 📋 Les resultats seront sauves dans vramancer_diagnostic.txt
echo.
pause
goto menu

:rapport_final
cls
echo ════════════ RAPPORT FINAL ════════════
echo.
echo 📊 Generation rapport de statut...
echo 📋 Resume complet avec recommandations
echo.
echo Appuyez sur une touche pour demarrer...
pause >nul
start "Rapport Final" cmd /c "cd /d "%~dp0" && vramancer_rapport.bat"
echo.
echo ✅ Rapport lance dans une nouvelle fenetre
echo 📋 Les resultats seront sauves dans vramancer_rapport_final.txt
echo.
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
echo ✅ Ports nettoyes
pause
goto menu

:status_systeme
cls
echo ════════════ STATUS SYSTEME ════════════
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
echo Ports utilises:
netstat -an | findstr ":5030 :8080 :5000 :5003" 2>nul
if errorlevel 1 echo Aucun port VRAMancer utilise
pause
goto menu

:cleanup_bat
cls
echo ════════════ NETTOYAGE ANCIENS .BAT ════════════
echo.
echo 🧹 Suppression des anciens fichiers .bat...
echo ⚠️  Suppression des fichiers obsoletes uniquement
echo.
set /p confirm="Confirmer suppression? (O/N): "
if /i not "%confirm%"=="O" goto menu

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
echo ✅ Anciens fichiers .bat supprimes
pause
goto menu

:liste_bat
cls
echo ════════════ LISTE FICHIERS .BAT ════════════
echo.
echo Fichiers .bat presents:
dir *.bat /b 2>nul
echo.
pause
goto menu

:fin
cls
echo.
echo 👋 Au revoir !
echo.
echo 📋 Rappel des fichiers principaux:
echo    - vramancer_menu_principal.bat (ce menu)
echo    - api_permanente.bat (API)
echo    - vramancer_test_complet.bat (tests)
echo    - vramancer_diagnostic.bat (diagnostic)
echo    - vramancer_rapport.bat (rapport)
echo.
pause
exit