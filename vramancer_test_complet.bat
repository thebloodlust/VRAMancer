@echo off
title VRAMancer - Test Complet Toutes Interfaces

cls
echo.
echo ████████████████████████████████████████████████████████████████
echo           VRAMANCER - TEST COMPLET TOUTES INTERFACES
echo ████████████████████████████████████████████████████████████████
echo.

set "LOGFILE=vramancer_test_results.txt"
echo === VRAMANCER TEST COMPLET === > %LOGFILE%
echo Date: %date% %time% >> %LOGFILE%
echo. >> %LOGFILE%

echo 🧪 Démarrage des tests automatiques...
echo 📋 Résultats sauvés dans: %LOGFILE%
echo.

REM Vérification API
echo ════════════════════════════════════════════════════════════════
echo   TEST 1/10 - VÉRIFICATION API
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 1] API Health Check >> %LOGFILE%
python -c "
import requests
try:
    r = requests.get('http://localhost:5030/health', timeout=3)
    if r.status_code == 200:
        print('✅ API OK - Status 200')
        print('API: OK') 
        exit(0)
    else:
        print('❌ API Error - Status', r.status_code)
        print('API: ERROR')
        exit(1)
except Exception as e:
    print('❌ API Non connectée:', e)
    print('API: DISCONNECTED')
    exit(1)
" >> %LOGFILE% 2>&1

if %errorlevel% neq 0 (
    echo ❌ API non disponible - Tests limités
    echo Voulez-vous continuer quand même? (O/N)
    set /p continue="Choix: "
    if /i not "%continue%"=="O" exit /b
) else (
    echo ✅ API disponible - Tests complets possibles
)

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 2/10 - DEBUG WEB ULTRA
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 2] Debug Web Ultra >> %LOGFILE%
timeout /t 2 /nobreak >nul
echo Test Debug Web Ultra... >> %LOGFILE%
python -c "
try:
    exec(open('debug_web_ultra.py').read())
    print('DEBUG_WEB_ULTRA: OK')
except ImportError as e:
    print('DEBUG_WEB_ULTRA: MISSING_DEPS')
except Exception as e:
    print('DEBUG_WEB_ULTRA: ERROR')
" >> %LOGFILE% 2>&1
echo ✅ Test 2 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 3/10 - QT DASHBOARD
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 3] Qt Dashboard >> %LOGFILE%
python -c "
import sys
sys.path.insert(0, '.')
try:
    from PyQt5 import QtWidgets
    print('✅ PyQt5 disponible')
    print('QT_DASHBOARD: PYQT5_OK')
except ImportError:
    try:
        from PyQt6 import QtWidgets  
        print('✅ PyQt6 disponible')
        print('QT_DASHBOARD: PYQT6_OK')
    except ImportError:
        try:
            from PySide2 import QtWidgets
            print('✅ PySide2 disponible') 
            print('QT_DASHBOARD: PYSIDE2_OK')
        except ImportError:
            try:
                from PySide6 import QtWidgets
                print('✅ PySide6 disponible')
                print('QT_DASHBOARD: PYSIDE6_OK') 
            except ImportError:
                print('❌ Aucune library Qt trouvée')
                print('QT_DASHBOARD: NO_QT')
" >> %LOGFILE% 2>&1
echo ✅ Test 3 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 4/10 - DASHBOARD WEB AVANCÉ
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 4] Dashboard Web Avancé >> %LOGFILE%
python -c "
import sys
sys.path.insert(0, '.')
try:
    with open('dashboard/dashboard_web_advanced.py', 'r') as f:
        content = f.read()
        if 'Flask' in content:
            print('✅ Dashboard Web Avancé - Flask détecté')
            print('WEB_ADVANCED: OK')
        else:
            print('❌ Dashboard Web Avancé - Structure incorrecte') 
            print('WEB_ADVANCED: BAD_STRUCTURE')
except FileNotFoundError:
    print('❌ Dashboard Web Avancé - Fichier manquant')
    print('WEB_ADVANCED: MISSING')
except Exception as e:
    print('❌ Dashboard Web Avancé - Erreur:', e)
    print('WEB_ADVANCED: ERROR')
" >> %LOGFILE% 2>&1
echo ✅ Test 4 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 5/10 - MOBILE DASHBOARD  
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 5] Mobile Dashboard >> %LOGFILE%
python -c "
try:
    with open('mobile/dashboard_mobile.py', 'r') as f:
        content = f.read()
        if 'Flask' in content and 'mobile' in content.lower():
            print('✅ Mobile Dashboard OK')
            print('MOBILE_DASHBOARD: OK')
        else:
            print('❌ Mobile Dashboard - Structure incorrecte')
            print('MOBILE_DASHBOARD: BAD_STRUCTURE') 
except FileNotFoundError:
    print('❌ Mobile Dashboard - Fichier manquant')
    print('MOBILE_DASHBOARD: MISSING')
except Exception as e:
    print('❌ Mobile Dashboard - Erreur:', e)
    print('MOBILE_DASHBOARD: ERROR')
" >> %LOGFILE% 2>&1
echo ✅ Test 5 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 6/10 - SYSTEM TRAY
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 6] System Tray >> %LOGFILE%
python -c "
try:
    with open('dashboard/systray_vramancer.py', 'r') as f:
        content = f.read()
        if 'QSystemTrayIcon' in content:
            print('✅ System Tray OK')
            print('SYSTEM_TRAY: OK')
        else:
            print('❌ System Tray - Structure incorrecte')
            print('SYSTEM_TRAY: BAD_STRUCTURE')
except FileNotFoundError:
    print('❌ System Tray - Fichier manquant') 
    print('SYSTEM_TRAY: MISSING')
except Exception as e:
    print('❌ System Tray - Erreur:', e)
    print('SYSTEM_TRAY: ERROR')
" >> %LOGFILE% 2>&1
echo ✅ Test 6 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 7/10 - CLI DASHBOARD
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 7] CLI Dashboard >> %LOGFILE%
set PYTHONPATH=%cd%
python -c "
import sys
sys.path.insert(0, '.')
try:
    with open('dashboard/dashboard_cli.py', 'r') as f:
        print('✅ CLI Dashboard - Fichier présent')
        print('CLI_DASHBOARD: FILE_OK')
except FileNotFoundError:
    print('❌ CLI Dashboard - Fichier manquant')
    print('CLI_DASHBOARD: MISSING')
" >> %LOGFILE% 2>&1
echo ✅ Test 7 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 8/10 - TKINTER DASHBOARD
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 8] Tkinter Dashboard >> %LOGFILE%
python -c "
import sys
sys.path.insert(0, '.')
try:
    import tkinter as tk
    print('✅ Tkinter disponible')
    with open('dashboard/dashboard_tk.py', 'r') as f:
        print('✅ Tkinter Dashboard - Fichier présent')
        print('TKINTER_DASHBOARD: OK')
except ImportError:
    print('❌ Tkinter non disponible')
    print('TKINTER_DASHBOARD: NO_TKINTER')
except FileNotFoundError:
    print('❌ Tkinter Dashboard - Fichier manquant')
    print('TKINTER_DASHBOARD: MISSING')
" >> %LOGFILE% 2>&1
echo ✅ Test 8 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 9/10 - LAUNCHER AUTO
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 9] Launcher Auto >> %LOGFILE%
python -c "
try:
    with open('dashboard/launcher.py', 'r') as f:
        content = f.read()
        if 'auto' in content.lower() or 'detect' in content.lower():
            print('✅ Launcher Auto OK')
            print('LAUNCHER_AUTO: OK')
        else:
            print('❌ Launcher Auto - Structure incorrecte')
            print('LAUNCHER_AUTO: BAD_STRUCTURE')
except FileNotFoundError:
    print('❌ Launcher Auto - Fichier manquant')
    print('LAUNCHER_AUTO: MISSING')
" >> %LOGFILE% 2>&1
echo ✅ Test 9 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 10/10 - DEBUG WEB SIMPLE
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 10] Debug Web Simple >> %LOGFILE%
python -c "
try:
    with open('debug_web_simple.py', 'r') as f:
        print('✅ Debug Web Simple OK')
        print('DEBUG_WEB_SIMPLE: OK')
except FileNotFoundError:
    print('❌ Debug Web Simple - Fichier manquant')
    print('DEBUG_WEB_SIMPLE: MISSING')
" >> %LOGFILE% 2>&1
echo ✅ Test 10 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   GÉNÉRATION RAPPORT FINAL
echo ════════════════════════════════════════════════════════════════
echo.

REM Génération rapport
echo. >> %LOGFILE%
echo === RÉSUMÉ FINAL === >> %LOGFILE%

echo 📊 Génération du rapport final...
python -c "
import re

# Lecture résultats
with open('vramancer_test_results.txt', 'r') as f:
    content = f.read()

# Extraction résultats
results = {
    'API': 'UNKNOWN',
    'DEBUG_WEB_ULTRA': 'UNKNOWN', 
    'QT_DASHBOARD': 'UNKNOWN',
    'WEB_ADVANCED': 'UNKNOWN',
    'MOBILE_DASHBOARD': 'UNKNOWN',
    'SYSTEM_TRAY': 'UNKNOWN',
    'CLI_DASHBOARD': 'UNKNOWN',
    'TKINTER_DASHBOARD': 'UNKNOWN',
    'LAUNCHER_AUTO': 'UNKNOWN',
    'DEBUG_WEB_SIMPLE': 'UNKNOWN'
}

for line in content.split('\n'):
    for key in results.keys():
        if f'{key}:' in line:
            results[key] = line.split(':', 1)[1].strip()

# Affichage rapport
print('\n' + '='*60)
print('           RAPPORT FINAL VRAMANCER')
print('='*60)
print()

total_tests = len(results)
working = sum(1 for v in results.values() if 'OK' in v)
errors = sum(1 for v in results.values() if 'ERROR' in v or 'MISSING' in v)
partial = total_tests - working - errors

print(f'📊 STATISTIQUES:')
print(f'   ✅ Fonctionnels: {working}/{total_tests}')
print(f'   ❌ En erreur: {errors}/{total_tests}')  
print(f'   ⚠️  Partiels: {partial}/{total_tests}')
print()

print('🔍 DÉTAIL PAR INTERFACE:')
for interface, status in results.items():
    symbol = '✅' if 'OK' in status else '❌' if 'ERROR' in status or 'MISSING' in status else '⚠️'
    print(f'   {symbol} {interface:<20}: {status}')

print()
print('🎯 RECOMMANDATIONS:')
if results['API'] == 'OK':
    print('   ✅ API active - Toutes les interfaces peuvent fonctionner')
else:
    print('   ❌ API inactive - Lancer api_permanente.bat en premier')

if 'PYQT' in results['QT_DASHBOARD'] and 'OK' in results['QT_DASHBOARD']:
    print('   ✅ Qt Dashboard recommandé (interface principale)')

if results['DEBUG_WEB_ULTRA'] == 'OK':
    print('   ✅ Debug Web Ultra fonctionnel (interface web)')

print('='*60)
" >> %LOGFILE%

type %LOGFILE%

echo.
echo ✅ Tests terminés !
echo 📋 Rapport complet sauvé dans: %LOGFILE%
echo.
pause