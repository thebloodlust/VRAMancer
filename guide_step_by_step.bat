@echo off
title VRAMancer - Guide Step-by-Step

echo ===============================================
echo   VRAMANCER - GUIDE STEP-BY-STEP
echo ===============================================
echo.
echo Ce guide vous accompagne pour tester toutes les interfaces
echo et diagnostiquer le probleme web bugge.
echo.

:step1
echo ===============================================
echo   ETAPE 1: VERIFICATION API
echo ===============================================
echo.
echo 1. L'API doit tourner en permanence
echo 2. Si pas encore fait, lancez dans une autre fenetre:
echo    api_permanente.bat
echo.
echo Test de l'API...
python -c "import requests; requests.get('http://localhost:5030/health', timeout=2)" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ API NON ACCESSIBLE
    echo.
    echo ACTIONS REQUISES:
    echo 1. Ouvrez une nouvelle fenetre CMD
    echo 2. Lancez: api_permanente.bat  
    echo 3. Attendez voir "API minimale demarree sur port 5030"
    echo 4. Revenez ici et appuyez sur une touche
    echo.
    pause
    goto step1
)
echo ✅ API accessible sur port 5030
echo.

:step2
echo ===============================================
echo   ETAPE 2: TEST INTERFACES SUPERVISION
echo ===============================================
echo.
echo On va tester dans l'ordre de probabilite de fonctionnement:
echo.
echo A. Dashboard Web Avance (supervision cluster)
echo B. Qt Dashboard (natif - deja teste)
echo C. System Tray (monitoring permanent)
echo D. Debug Web Bugge (problematique)
echo E. Debug Web Simple (diagnostic)
echo.
set /p ready=Pret pour les tests? (o/N): 

if /i not "%ready%"=="o" if /i not "%ready%"=="oui" goto end

:test_web_advanced
echo.
echo ===============================================
echo   TEST A: DASHBOARD WEB AVANCE
echo ===============================================
echo.
echo Interface de supervision cluster temps reel
echo Port: 8080 (different du debug web bugge)
echo.
echo Lancement...
if exist "dashboard\dashboard_web_advanced.py" (
    echo Interface sur http://localhost:8080
    echo Fermez avec Ctrl+C pour continuer les tests
    echo.
    python dashboard\dashboard_web_advanced.py
    echo.
    echo Dashboard Web Avance termine.
    echo.
    set /p worked=Cette interface a-t-elle fonctionne? (o/N): 
    if /i "%worked%"=="o" echo ✅ Dashboard Web Avance: FONCTIONNE
    if /i "%worked%"=="oui" echo ✅ Dashboard Web Avance: FONCTIONNE
    if /i not "%worked%"=="o" if /i not "%worked%"=="oui" echo ❌ Dashboard Web Avance: PROBLEME
) else (
    echo ❌ Fichier dashboard\dashboard_web_advanced.py non trouve
)
echo.

:test_systray
echo ===============================================
echo   TEST B: SYSTEM TRAY
echo ===============================================
echo.
echo Interface systray avec monitoring permanent
echo (Si ca marche, une icone apparait dans la barre des taches)
echo.
set /p test_systray=Tester System Tray? (o/N): 
if /i not "%test_systray%"=="o" if /i not "%test_systray%"=="oui" goto test_debug_buggy

echo Lancement System Tray...
if exist "systray_vramancer.py" (
    echo Regardez la barre des taches pour l'icone VRAMancer
    echo Fermez en cliquant droit sur l'icone puis Quitter
    echo.
    python systray_vramancer.py
    echo.
    echo System Tray termine.
    echo.
    set /p worked2=System Tray a-t-il fonctionne? (o/N): 
    if /i "%worked2%"=="o" echo ✅ System Tray: FONCTIONNE
    if /i "%worked2%"=="oui" echo ✅ System Tray: FONCTIONNE
    if /i not "%worked2%"=="o" if /i not "%worked2%"=="oui" echo ❌ System Tray: PROBLEME
) else (
    echo ❌ Fichier systray_vramancer.py non trouve
)
echo.

:test_debug_buggy
echo ===============================================
echo   TEST C: DEBUG WEB BUGGE (PROBLEMATIQUE)
echo ===============================================
echo.
echo Maintenant testons l'interface qui pose probleme
echo Celle qui reste bloquee sur "Verification..."
echo.
set /p test_buggy=Tester Debug Web bugge? (o/N): 
if /i not "%test_buggy%"=="o" if /i not "%test_buggy%"=="oui" goto test_debug_simple

echo Lancement Debug Web (version problematique)...
echo Interface sur http://localhost:8080
echo ATTENDU: Interface reste bloquee sur "Verification..."
echo.
echo Fermez avec Ctrl+C quand vous avez confirme le probleme
echo.
python debug_web.py
echo.
echo Debug Web termine.
echo.

:test_debug_simple
echo ===============================================
echo   TEST D: DEBUG WEB SIMPLE (DIAGNOSTIC)
echo ===============================================
echo.
echo Version simplifiee pour confirmer que le probleme 
echo n'est pas dans l'API mais dans l'interface complexe
echo.
echo Lancement Debug Web Simple...
echo Interface sur http://localhost:8080
echo ATTENDU: Boutons fonctionnels, tests API reussis
echo.
echo Fermez avec Ctrl+C pour terminer
echo.
python debug_web_simple.py
echo.
echo Debug Web Simple termine.
echo.

:results
echo ===============================================
echo   RESULTATS DES TESTS
echo ===============================================
echo.
echo Interfaces testees:
echo - Dashboard Web Avance: Interface supervision cluster
echo - System Tray: Monitoring permanent barre des taches  
echo - Debug Web Bugge: Interface problematique
echo - Debug Web Simple: Version diagnostic qui marche
echo.
echo CONCLUSION:
echo - Si Web Avance marche: API OK, probleme dans debug_web.py
echo - Si System Tray marche: Qt OK, probleme JavaScript debug web
echo - Si Simple marche: Confirme que probleme = interface complexe
echo.
echo Le bug est probablement dans le JavaScript de debug_web.py
echo qui ne met pas a jour le statut API correctement.
echo.

:end
echo Fin du guide step-by-step
pause