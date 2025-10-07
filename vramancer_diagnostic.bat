@echo off
title VRAMancer - Diagnostic Complet

cls
echo.
echo ████████████████████████████████████████████████████████████████
echo              VRAMANCER - DIAGNOSTIC COMPLET
echo ████████████████████████████████████████████████████████████════
echo.

set "DIAGFILE=vramancer_diagnostic.txt"
echo === DIAGNOSTIC VRAMANCER === > %DIAGFILE%
echo Date: %date% %time% >> %DIAGFILE%
echo. >> %DIAGFILE%

echo 🔍 Diagnostic système complet en cours...
echo 📋 Résultats dans: %DIAGFILE%
echo.

echo ════════════════════════════════════════════════════════════════
echo   DIAGNOSTIC 1/8 - ENVIRONNEMENT PYTHON
echo ════════════════════════════════════════════════════════════════
echo.
echo [DIAG 1] Environnement Python >> %DIAGFILE%
python -c "
import sys, os
print(f'Python Version: {sys.version}')
print(f'Python Executable: {sys.executable}')
print(f'Working Directory: {os.getcwd()}')
print(f'Python Path: {sys.path[:3]}...')
print()

# Test imports critiques
imports_test = [
    'requests', 'flask', 'tkinter', 'json', 'os', 'sys',
    'threading', 'subprocess', 'datetime', 'traceback'
]

print('=== IMPORTS CRITIQUES ===')
for imp in imports_test:
    try:
        __import__(imp)
        print(f'✅ {imp}: OK')
    except ImportError:
        print(f'❌ {imp}: MISSING')
print()

# Test imports Qt
print('=== IMPORTS QT ===') 
for qt_lib in ['PyQt5.QtWidgets', 'PyQt6.QtWidgets', 'PySide2.QtWidgets', 'PySide6.QtWidgets']:
    try:
        __import__(qt_lib)
        print(f'✅ {qt_lib}: OK')
        break
    except ImportError:
        print(f'❌ {qt_lib}: MISSING')
" >> %DIAGFILE% 2>&1
echo ✅ Diagnostic 1 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   DIAGNOSTIC 2/8 - STRUCTURE FICHIERS
echo ════════════════════════════════════════════════════════════════
echo.
echo [DIAG 2] Structure fichiers >> %DIAGFILE%
python -c "
import os

fichiers_critiques = [
    'start_api.py',
    'debug_web_ultra.py', 
    'debug_web_simple.py',
    'dashboard/dashboard_qt.py',
    'dashboard/dashboard_web_advanced.py',
    'dashboard/dashboard_cli.py',
    'dashboard/dashboard_tk.py',
    'dashboard/systray_vramancer.py',
    'dashboard/launcher.py',
    'mobile/dashboard_mobile.py',
    'api_permanente.bat'
]

print('=== FICHIERS CRITIQUES ===')
for fichier in fichiers_critiques:
    if os.path.exists(fichier):
        size = os.path.getsize(fichier)
        print(f'✅ {fichier:<35}: {size:>8} bytes')
    else:
        print(f'❌ {fichier:<35}: MISSING')

print()
print('=== DOSSIERS STRUCTURE ===')
dossiers = ['core', 'dashboard', 'mobile', 'cli', 'utils', 'docs']
for dossier in dossiers:
    if os.path.exists(dossier):
        files = len([f for f in os.listdir(dossier) if f.endswith('.py')])
        print(f'✅ {dossier:<15}: {files} fichiers Python')
    else:
        print(f'❌ {dossier:<15}: MISSING')
" >> %DIAGFILE% 2>&1
echo ✅ Diagnostic 2 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   DIAGNOSTIC 3/8 - RÉSEAU ET PORTS
echo ════════════════════════════════════════════════════════════════
echo.
echo [DIAG 3] Réseau et ports >> %DIAGFILE%
echo === PORTS UTILISÉS === >> %DIAGFILE%
netstat -an | findstr ":5030 :8080 :5000 :5003" >> %DIAGFILE%
echo. >> %DIAGFILE%

echo === TEST CONNECTIVITÉ === >> %DIAGFILE%
python -c "
import socket

ports_test = [5030, 8080, 5000, 5003]
print('=== TEST PORTS ===')
for port in ports_test:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('localhost', port))
    if result == 0:
        print(f'✅ Port {port}: OUVERT')
    else:
        print(f'❌ Port {port}: FERMÉ')
    sock.close()

print()
print('=== TEST API ENDPOINTS ===')
try:
    import requests
    endpoints = ['/health', '/api/status', '/api/gpu/info', '/api/nodes']
    for endpoint in endpoints:
        try:
            r = requests.get(f'http://localhost:5030{endpoint}', timeout=2)
            print(f'✅ {endpoint}: Status {r.status_code}')
        except Exception as e:
            print(f'❌ {endpoint}: {str(e)[:50]}')
except ImportError:
    print('❌ requests non disponible pour test API')
" >> %DIAGFILE% 2>&1
echo ✅ Diagnostic 3 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   DIAGNOSTIC 4/8 - PROCESSUS VRAMANCER
echo ════════════════════════════════════════════════════════════════
echo.
echo [DIAG 4] Processus VRAMancer >> %DIAGFILE%
echo === PROCESSUS PYTHON === >> %DIAGFILE%
tasklist | findstr python >> %DIAGFILE%
echo. >> %DIAGFILE%

echo ✅ Diagnostic 4 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   DIAGNOSTIC 5/8 - LOGS ET ERREURS
echo ════════════════════════════════════════════════════════════════
echo.
echo [DIAG 5] Logs et erreurs >> %DIAGFILE%
echo === RECHERCHE LOGS === >> %DIAGFILE%
dir *.log /b 2>nul >> %DIAGFILE%
dir *error* /b 2>nul >> %DIAGFILE%
echo. >> %DIAGFILE%
echo ✅ Diagnostic 5 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   DIAGNOSTIC 6/8 - CONFIGURATION SYSTÈME
echo ════════════════════════════════════════════════════════════════
echo.
echo [DIAG 6] Configuration système >> %DIAGFILE%
echo === VARIABLES ENVIRONNEMENT === >> %DIAGFILE%
set | findstr VRM >> %DIAGFILE%
set | findstr PYTHON >> %DIAGFILE%
echo. >> %DIAGFILE%
echo ✅ Diagnostic 6 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   DIAGNOSTIC 7/8 - MÉMOIRE ET PERFORMANCE
echo ════════════════════════════════════════════════════════════════
echo.
echo [DIAG 7] Mémoire et performance >> %DIAGFILE%
python -c "
import psutil
import os

print('=== SYSTÈME ===')
print(f'CPU Usage: {psutil.cpu_percent()}%')
print(f'Memory Usage: {psutil.virtual_memory().percent}%')
print(f'Disk Usage: {psutil.disk_usage(\".\" if os.name == \"posix\" else \"C:\\\\\").percent}%')
print()

print('=== PROCESSUS PYTHON ===')
for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
    if 'python' in proc.info['name'].lower():
        print(f'PID {proc.info[\"pid\"]}: CPU {proc.info[\"cpu_percent\"]}% MEM {proc.info[\"memory_percent\"]}%')
" >> %DIAGFILE% 2>&1
echo ✅ Diagnostic 7 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   DIAGNOSTIC 8/8 - RECOMMANDATIONS
echo ════════════════════════════════════════════════════════════════
echo.
echo [DIAG 8] Génération recommandations >> %DIAGFILE%
python -c "
print()
print('=== RECOMMANDATIONS AUTOMATIQUES ===')
print()

# Lecture du diagnostic
try:
    with open('vramancer_diagnostic.txt', 'r') as f:
        content = f.read()
    
    recommendations = []
    
    # Check API
    if ':5030' not in content:
        recommendations.append('🔧 Lancer API permanente: api_permanente.bat')
    else:
        recommendations.append('✅ API détectée sur port 5030')
    
    # Check Qt
    if 'PyQt' in content and 'OK' in content:
        recommendations.append('✅ Qt disponible - Utiliser Qt Dashboard')
    else:
        recommendations.append('⚠️ Qt manquant - Installer PyQt5/6 ou PySide2/6')
    
    # Check Flask
    if 'flask: OK' in content:
        recommendations.append('✅ Flask OK - Interfaces web disponibles')
    else:
        recommendations.append('🔧 Installer Flask: pip install flask')
    
    # Check structure
    if 'debug_web_ultra.py' in content and 'bytes' in content:
        recommendations.append('✅ Debug Web Ultra disponible')
    else:
        recommendations.append('⚠️ Fichiers debug manquants - Vérifier repo')
    
    # Usage recommendations
    recommendations.append('')
    recommendations.append('📋 ORDRE D\'UTILISATION RECOMMANDÉ:')
    recommendations.append('1. Lancer api_permanente.bat')
    recommendations.append('2. Utiliser Qt Dashboard (interface principale)')
    recommendations.append('3. Debug Web Ultra pour diagnostic web')
    recommendations.append('4. System Tray pour monitoring permanent')
    
    for rec in recommendations:
        print(rec)
        
except Exception as e:
    print(f'Erreur génération recommandations: {e}')
" >> %DIAGFILE% 2>&1
echo ✅ Diagnostic 8 terminé

echo.
echo ════════════════════════════════════════════════════════════════
echo   AFFICHAGE RAPPORT FINAL
echo ════════════════════════════════════════════════════════════════
echo.

type %DIAGFILE%

echo.
echo ✅ Diagnostic complet terminé !
echo 📋 Rapport détaillé sauvé dans: %DIAGFILE%
echo.
pause