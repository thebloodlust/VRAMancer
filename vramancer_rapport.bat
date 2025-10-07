@echo off
title VRAMancer - Rapport Final

cls
echo.
echo ████████████████████████████████████████████████████████████████
echo                 VRAMANCER - RAPPORT FINAL
echo ████████████████████████████████████████████████████████████████
echo.

set "RAPPORTFILE=vramancer_rapport_final.txt"
echo === RAPPORT FINAL VRAMANCER === > %RAPPORTFILE%
echo Date: %date% %time% >> %RAPPORTFILE%
echo. >> %RAPPORTFILE%

echo 📊 Génération du rapport final...
echo 📋 Sauvegarde dans: %RAPPORTFILE%
echo.

REM Collecte des informations
python -c "
import os
import sys
from datetime import datetime

def test_interface(file_path, name):
    '''Test si une interface est disponible'''
    try:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            return f'✅ {name}: DISPONIBLE ({size} bytes)'
        else:
            return f'❌ {name}: FICHIER MANQUANT'
    except Exception as e:
        return f'⚠️ {name}: ERREUR ({str(e)[:30]})'

def test_dependency(module_name):
    '''Test si une dépendance est disponible'''
    try:
        __import__(module_name)
        return f'✅ {module_name}: OK'
    except ImportError:
        return f'❌ {module_name}: MANQUANT'

def test_api():
    '''Test connectivité API'''
    try:
        import requests
        r = requests.get('http://localhost:5030/health', timeout=2)
        if r.status_code == 200:
            return '✅ API: ACTIVE (port 5030)'
        else:
            return f'⚠️ API: ERREUR (status {r.status_code})'
    except Exception as e:
        return '❌ API: NON CONNECTÉE'

print('=' * 80)
print('                    VRAMANCER - RAPPORT FINAL')
print('=' * 80)
print(f'Généré le: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
print(f'Répertoire: {os.getcwd()}')
print(f'Python: {sys.version.split()[0]}')
print()

# Test API
print('🔌 INFRASTRUCTURE:')
api_status = test_api()
print(f'   {api_status}')
print()

# Test dépendances
print('📦 DÉPENDANCES:')
deps = ['requests', 'flask', 'tkinter', 'json', 'threading']
for dep in deps:
    status = test_dependency(dep)
    print(f'   {status}')
print()

# Test Qt
print('🖥️ LIBRAIRIES QT:')
qt_libs = ['PyQt5.QtWidgets', 'PyQt6.QtWidgets', 'PySide2.QtWidgets', 'PySide6.QtWidgets']
qt_found = False
for qt_lib in qt_libs:
    try:
        __import__(qt_lib)
        print(f'   ✅ {qt_lib}: OK')
        qt_found = True
        break
    except ImportError:
        continue
if not qt_found:
    print('   ❌ Aucune librairie Qt trouvée')
print()

# Test interfaces
print('🎛️ INTERFACES DISPONIBLES:')
interfaces = [
    ('debug_web_ultra.py', 'Debug Web Ultra'),
    ('debug_web_simple.py', 'Debug Web Simple'),
    ('dashboard/dashboard_qt.py', 'Qt Dashboard'),
    ('dashboard/dashboard_web_advanced.py', 'Dashboard Web Avancé'),
    ('dashboard/dashboard_cli.py', 'CLI Dashboard'),
    ('dashboard/dashboard_tk.py', 'Tkinter Dashboard'),
    ('dashboard/systray_vramancer.py', 'System Tray'),
    ('dashboard/launcher.py', 'Launcher Auto'),
    ('mobile/dashboard_mobile.py', 'Mobile Dashboard')
]

interface_status = {}
for file_path, name in interfaces:
    status = test_interface(file_path, name)
    interface_status[name] = status
    print(f'   {status}')
print()

# Test fichiers de lancement
print('🚀 LANCEURS:')
lanceurs = [
    ('api_permanente.bat', 'API Permanente'),
    ('vramancer_menu_principal.bat', 'Menu Principal'),
    ('vramancer_test_complet.bat', 'Test Complet'),
    ('vramancer_diagnostic.bat', 'Diagnostic'),
]
for file_path, name in lanceurs:
    status = test_interface(file_path, name)
    print(f'   {status}')
print()

# Statistiques
total_interfaces = len(interfaces)
working_interfaces = sum(1 for status in interface_status.values() if '✅' in status)
missing_interfaces = sum(1 for status in interface_status.values() if '❌' in status)

print('📊 STATISTIQUES:')
print(f'   Total interfaces: {total_interfaces}')
print(f'   Fonctionnelles: {working_interfaces} ({working_interfaces/total_interfaces*100:.1f}%)')
print(f'   Manquantes: {missing_interfaces} ({missing_interfaces/total_interfaces*100:.1f}%)')
print()

# Recommandations
print('🎯 RECOMMANDATIONS:')
if '✅ API: ACTIVE' in api_status:
    print('   ✅ API active - Système prêt à l\'usage')
    print('   📋 Interface recommandée: Qt Dashboard (stable)')
    print('   🌐 Interface web: Debug Web Ultra (diagnostics)')
    print('   🪟 Monitoring: System Tray (permanent)')
else:
    print('   🔧 PRIORITÉ 1: Lancer api_permanente.bat')
    print('   ⚠️ Toutes les interfaces nécessitent l\'API active')

if qt_found:
    print('   ✅ Qt disponible - Qt Dashboard utilisable')
else:
    print('   🔧 Installer Qt: pip install PyQt5 ou PyQt6')

print('   📚 Guide: vramancer_menu_principal.bat (menu unifié)')
print('   🧪 Tests: vramancer_test_complet.bat (validation)')
print('   🔍 Debug: vramancer_diagnostic.bat (problèmes)')
print()

print('🏁 PROCHAINES ÉTAPES:')
print('   1. Lancer vramancer_menu_principal.bat')
print('   2. Choisir option 1 (API Permanente) si pas déjà fait')
print('   3. Choisir option 11 (Qt Dashboard) ou 10 (Debug Web Ultra)')
print('   4. Pour problèmes: option 31 (Diagnostic Complet)')
print()

print('=' * 80)
print('                   RAPPORT TERMINÉ')
print('=' * 80)
" >> %RAPPORTFILE% 2>&1

type %RAPPORTFILE%

echo.
echo ✅ Rapport final généré !
echo 📋 Sauvé dans: %RAPPORTFILE%
echo.
echo 🎯 ÉTAPES SUIVANTES:
echo    1. Lancer vramancer_menu_principal.bat
echo    2. Suivre les recommandations du rapport
echo.
pause