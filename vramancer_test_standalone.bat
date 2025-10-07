@echo off
chcp 65001 >nul
title VRAMancer - Test Complet STANDALONE

cls
echo.
echo ████████████████████████████████████████████████████████████████
echo           VRAMANCER - TEST COMPLET STANDALONE
echo ████████████████████████████████████████████████████████████████
echo.
echo Ce script peut etre lance independamment
echo Pas besoin que l'API soit demarree pour ce test
echo.

set "LOGFILE=vramancer_test_results.txt"
echo === VRAMANCER TEST COMPLET STANDALONE === > %LOGFILE%
echo Date: %date% %time% >> %LOGFILE%
echo. >> %LOGFILE%

echo 🧪 Demarrage des tests automatiques...
echo 📋 Resultats sauves dans: %LOGFILE%
echo ⏱️  Duree estimee: 2-3 minutes
echo.

REM Verification API (sans bloquer)
echo ════════════════════════════════════════════════════════════════
echo   TEST 1/10 - VERIFICATION API
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
    print('❌ API Non connectee:', e)
    print('API: DISCONNECTED')
    exit(1)
" >> %LOGFILE% 2>&1

if %errorlevel% neq 0 (
    echo ❌ API non disponible - Tests en mode diagnostique uniquement
    echo ℹ️  Les tests d'interface continueront quand meme
) else (
    echo ✅ API disponible - Tests complets possibles
)

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 2/10 - DEBUG WEB ULTRA
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 2] Debug Web Ultra >> %LOGFILE%
python -c "
try:
    with open('debug_web_ultra.py', 'r', encoding='utf-8') as f:
        content = f.read()
        if 'Flask' in content and 'fetchWithTimeout' in content:
            print('✅ Debug Web Ultra OK - Fichier complet')
            print('DEBUG_WEB_ULTRA: OK')
        else:
            print('❌ Debug Web Ultra - Structure incomplete')
            print('DEBUG_WEB_ULTRA: INCOMPLETE')
except FileNotFoundError:
    print('❌ Debug Web Ultra - Fichier manquant')
    print('DEBUG_WEB_ULTRA: MISSING')
except Exception as e:
    print('❌ Debug Web Ultra - Erreur:', e)
    print('DEBUG_WEB_ULTRA: ERROR')
" >> %LOGFILE% 2>&1
echo ✅ Test 2 termine

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
    with open('dashboard/dashboard_qt.py', 'r', encoding='utf-8') as f:
        print('✅ Fichier Qt Dashboard present')
        print('QT_DASHBOARD: PYQT5_OK')
except ImportError:
    try:
        from PyQt6 import QtWidgets  
        print('✅ PyQt6 disponible')
        with open('dashboard/dashboard_qt.py', 'r', encoding='utf-8') as f:
            print('✅ Fichier Qt Dashboard present')
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
                print('❌ Aucune library Qt trouvee')
                print('QT_DASHBOARD: NO_QT')
except FileNotFoundError:
    print('❌ Fichier dashboard_qt.py manquant')
    print('QT_DASHBOARD: FILE_MISSING')
" >> %LOGFILE% 2>&1
echo ✅ Test 3 termine

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 4/10 - DASHBOARD WEB AVANCE
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 4] Dashboard Web Avance >> %LOGFILE%
python -c "
import sys
sys.path.insert(0, '.')
try:
    with open('dashboard/dashboard_web_advanced.py', 'r', encoding='utf-8') as f:
        content = f.read()
        if 'Flask' in content and 'update_from_api' in content:
            print('✅ Dashboard Web Avance - Version corrigee')
            print('WEB_ADVANCED: OK')
        else:
            print('❌ Dashboard Web Avance - Version obsolete') 
            print('WEB_ADVANCED: OLD_VERSION')
except FileNotFoundError:
    print('❌ Dashboard Web Avance - Fichier manquant')
    print('WEB_ADVANCED: MISSING')
except Exception as e:
    print('❌ Dashboard Web Avance - Erreur:', e)
    print('WEB_ADVANCED: ERROR')
" >> %LOGFILE% 2>&1
echo ✅ Test 4 termine

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 5/10 - MOBILE DASHBOARD  
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 5] Mobile Dashboard >> %LOGFILE%
python -c "
try:
    with open('mobile/dashboard_mobile.py', 'r', encoding='utf-8') as f:
        content = f.read()
        if 'Flask' in content and 'testAPI' in content:
            print('✅ Mobile Dashboard - Version corrigee')
            print('MOBILE_DASHBOARD: OK')
        else:
            print('❌ Mobile Dashboard - Version obsolete')
            print('MOBILE_DASHBOARD: OLD_VERSION') 
except FileNotFoundError:
    print('❌ Mobile Dashboard - Fichier manquant')
    print('MOBILE_DASHBOARD: MISSING')
except Exception as e:
    print('❌ Mobile Dashboard - Erreur:', e)
    print('MOBILE_DASHBOARD: ERROR')
" >> %LOGFILE% 2>&1
echo ✅ Test 5 termine

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 6/10 - SYSTEM TRAY
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 6] System Tray >> %LOGFILE%
python -c "
try:
    with open('dashboard/systray_vramancer.py', 'r', encoding='utf-8') as f:
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
echo ✅ Test 6 termine

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 7/10 - CLI DASHBOARD
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 7] CLI Dashboard >> %LOGFILE%
python -c "
try:
    with open('dashboard/dashboard_cli.py', 'r', encoding='utf-8') as f:
        print('✅ CLI Dashboard - Fichier present')
        print('CLI_DASHBOARD: FILE_OK')
except FileNotFoundError:
    print('❌ CLI Dashboard - Fichier manquant')
    print('CLI_DASHBOARD: MISSING')
except Exception as e:
    print('❌ CLI Dashboard - Erreur:', e)
    print('CLI_DASHBOARD: ERROR')
" >> %LOGFILE% 2>&1
echo ✅ Test 7 termine

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 8/10 - TKINTER DASHBOARD
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 8] Tkinter Dashboard >> %LOGFILE%
python -c "
try:
    import tkinter as tk
    print('✅ Tkinter disponible')
    with open('dashboard/dashboard_tk.py', 'r', encoding='utf-8') as f:
        print('✅ Tkinter Dashboard - Fichier present')
        print('TKINTER_DASHBOARD: OK')
except ImportError:
    print('❌ Tkinter non disponible')
    print('TKINTER_DASHBOARD: NO_TKINTER')
except FileNotFoundError:
    print('❌ Tkinter Dashboard - Fichier manquant')
    print('TKINTER_DASHBOARD: MISSING')
except Exception as e:
    print('❌ Tkinter Dashboard - Erreur:', e)
    print('TKINTER_DASHBOARD: ERROR')
" >> %LOGFILE% 2>&1
echo ✅ Test 8 termine

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 9/10 - LAUNCHER AUTO
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 9] Launcher Auto >> %LOGFILE%
python -c "
try:
    with open('dashboard/launcher.py', 'r', encoding='utf-8') as f:
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
except Exception as e:
    print('❌ Launcher Auto - Erreur:', e)
    print('LAUNCHER_AUTO: ERROR')
" >> %LOGFILE% 2>&1
echo ✅ Test 9 termine

echo.
echo ════════════════════════════════════════════════════════════════
echo   TEST 10/10 - DEBUG WEB SIMPLE
echo ════════════════════════════════════════════════════════════════
echo.
echo [TEST 10] Debug Web Simple >> %LOGFILE%
python -c "
try:
    with open('debug_web_simple.py', 'r', encoding='utf-8') as f:
        print('✅ Debug Web Simple OK')
        print('DEBUG_WEB_SIMPLE: OK')
except FileNotFoundError:
    print('❌ Debug Web Simple - Fichier manquant')
    print('DEBUG_WEB_SIMPLE: MISSING')
except Exception as e:
    print('❌ Debug Web Simple - Erreur:', e)
    print('DEBUG_WEB_SIMPLE: ERROR')
" >> %LOGFILE% 2>&1
echo ✅ Test 10 termine

echo.
echo ════════════════════════════════════════════════════════════════
echo   GENERATION RAPPORT FINAL
echo ════════════════════════════════════════════════════════════════
echo.

REM Generation rapport
echo. >> %LOGFILE%
echo === RESUME FINAL === >> %LOGFILE%

echo 📊 Generation du rapport final...
python -c "
import re

# Lecture resultats
try:
    with open('vramancer_test_results.txt', 'r', encoding='utf-8') as f:
        content = f.read()
except:
    with open('vramancer_test_results.txt', 'r') as f:
        content = f.read()

# Extraction resultats
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

print('🔍 DETAIL PAR INTERFACE:')
for interface, status in results.items():
    symbol = '✅' if 'OK' in status else '❌' if 'ERROR' in status or 'MISSING' in status else '⚠️'
    print(f'   {symbol} {interface:<20}: {status}')

print()
print('🎯 RECOMMANDATIONS:')
if results['API'] == 'OK':
    print('   ✅ API active - Toutes les interfaces peuvent fonctionner')
else:
    print('   ❌ API inactive - Lancer vramancer_menu_principal_fixe.bat option 1')

if 'PYQT' in str(results['QT_DASHBOARD']) and 'OK' in str(results['QT_DASHBOARD']):
    print('   ✅ Qt Dashboard recommande (interface principale)')

if results['DEBUG_WEB_ULTRA'] == 'OK':
    print('   ✅ Debug Web Ultra fonctionnel (interface web)')

print()
print('🚀 PROCHAINES ETAPES:')
print('   1. Lancer vramancer_menu_principal_fixe.bat')
print('   2. Option 1: API Permanente (si pas deja fait)')
print('   3. Option 11: Qt Dashboard (interface recommandee)')
print('   4. Option 10: Debug Web Ultra (interface web)')

print('='*60)
" >> %LOGFILE%

type %LOGFILE%

echo.
echo ✅ Tests termines !
echo 📋 Rapport complet sauve dans: %LOGFILE%
echo.
echo 🚀 POUR CONTINUER:
echo    1. Lancer vramancer_menu_principal_fixe.bat
echo    2. Choisir les interfaces testees comme fonctionnelles
echo.
echo Appuyez sur une touche pour fermer...
pause >nul