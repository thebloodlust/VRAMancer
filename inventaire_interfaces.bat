@echo off
title VRAMancer - Inventaire Interfaces

echo ===============================================
echo   VRAMANCER - INVENTAIRE COMPLET INTERFACES
echo ===============================================
echo.

echo Verification des fichiers d'interface...
echo.

echo === INTERFACES PRINCIPALES ===
if exist "dashboard\dashboard_qt.py" (echo ✅ Qt Dashboard - Interface native Qt) else (echo ❌ Qt Dashboard MANQUANT)
if exist "dashboard\dashboard_web.py" (echo ✅ Web Dashboard - Interface web standard) else (echo ❌ Web Dashboard MANQUANT)
if exist "dashboard\dashboard_tk.py" (echo ✅ Tk Dashboard - Interface Tkinter) else (echo ❌ Tk Dashboard MANQUANT)
if exist "debug_web.py" (echo ✅ Debug Web - Interface diagnostic) else (echo ❌ Debug Web MANQUANT)

echo.
echo === INTERFACES AVANCEES ===
if exist "dashboard\dashboard_web_advanced.py" (echo ✅ Web Advanced - Supervision cluster) else (echo ❌ Web Advanced MANQUANT)
if exist "dashboard\dashboard_cli.py" (echo ✅ CLI Dashboard - Interface ligne commande) else (echo ❌ CLI Dashboard MANQUANT)
if exist "systray_vramancer.py" (echo ✅ System Tray - Interface systray) else (echo ❌ System Tray MANQUANT)
if exist "dashboard\launcher.py" (echo ✅ Launcher - Auto-detection interface) else (echo ❌ Launcher MANQUANT)

echo.
echo === INTERFACES MOBILES ===
if exist "mobile\dashboard_mobile.py" (echo ✅ Mobile Dashboard - Interface mobile) else (echo ❌ Mobile Dashboard MANQUANT)
if exist "mobile\dashboard_heterogeneous.py" (echo ✅ Heterogeneous Mobile - Clusters hetero) else (echo ❌ Heterogeneous Mobile MANQUANT)

echo.
echo === INTERFACES DEBUG ===
if exist "debug_web_simple.py" (echo ✅ Debug Web Simple - Tests basiques) else (echo ❌ Debug Web Simple MANQUANT)

echo.
echo === OUTILS SUPERVISION ===
if exist "core\network\supervision.py" (echo ✅ Supervision Core - Monitoring backend) else (echo ❌ Supervision Core MANQUANT)
if exist "core\network\supervision_api.py" (echo ✅ Supervision API - API monitoring) else (echo ❌ Supervision API MANQUANT)
if exist "core\monitor.py" (echo ✅ Monitor Core - Monitoring GPU) else (echo ❌ Monitor Core MANQUANT)

echo.
echo ===============================================
echo   RESUME DES INTERFACES SUPERVISION
echo ===============================================
echo.
echo VRAMancer dispose de plusieurs niveaux d'interface:
echo.
echo 🔸 NIVEAU 1 - Basique:
echo   • Qt Dashboard: Interface native pour usage quotidien
echo   • Debug Web: Diagnostic et tests de connectivite
echo.
echo 🔸 NIVEAU 2 - Avance:
echo   • Web Advanced: Supervision cluster temps reel
echo   • System Tray: Monitoring permanent en arriere-plan
echo   • CLI Dashboard: Interface ligne de commande pour serveurs
echo.
echo 🔸 NIVEAU 3 - Specialise:
echo   • Mobile Dashboard: Interface adaptee tablettes/mobiles
echo   • Heterogeneous: Gestion clusters multi-architectures
echo   • Launcher: Auto-detection de la meilleure interface
echo.
echo Toutes ces interfaces utilisent la meme API sur port 5030
echo.

pause