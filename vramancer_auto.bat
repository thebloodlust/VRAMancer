@echo off
title VRAMancer - Lancement Automatique Complet

cls
echo ===============================================================================
echo                    VRAMANCER - LANCEMENT AUTOMATIQUE
echo ===============================================================================
echo.
echo Ce script verifie et lance l'API automatiquement
echo puis propose les interfaces disponibles
echo.

REM Test Python disponible
python --version >nul 2>&1 || (
    echo ERREUR: Python non trouve
    echo Installez Python depuis https://python.org
    pause
    exit /b 1
)

echo Verification API VRAMancer...
python check_api.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo ===============================================================================
    echo                        INTERFACES DISPONIBLES
    echo ===============================================================================
    echo.
    echo L'API est active - Choisissez votre interface:
    echo.
    echo 1. Qt Dashboard            (interface native recommandee)
    echo 2. Debug Web Ultra         (interface web complete)
    echo 3. Dashboard Web Avance    (supervision cluster)
    echo 4. Mobile Dashboard        (interface mobile)
    echo 5. System Tray             (monitoring permanent)
    echo 6. Menu complet            (toutes les options)
    echo.
    echo 9. Quitter
    echo.
    
    :choice_loop
    set /p choice="Votre choix (1-9): "
    
    if "%choice%"=="1" (
        echo Lancement Qt Dashboard...
        python dashboard\dashboard_qt.py
    ) else if "%choice%"=="2" (
        echo Lancement Debug Web Ultra...
        python debug_web_ultra.py
    ) else if "%choice%"=="3" (
        echo Lancement Dashboard Web Avance...
        python dashboard\dashboard_web_advanced.py
    ) else if "%choice%"=="4" (
        echo Lancement Mobile Dashboard...
        python mobile\dashboard_mobile.py
    ) else if "%choice%"=="5" (
        echo Lancement System Tray...
        python systray_vramancer.py
    ) else if "%choice%"=="6" (
        echo Lancement menu complet...
        call vramancer_menu_simple.bat
    ) else if "%choice%"=="9" (
        exit /b 0
    ) else (
        echo Choix invalide, essayez encore.
        goto choice_loop
    )
    
    echo.
    echo Interface fermee. Relancer une autre interface?
    goto choice_loop
    
) else (
    echo.
    echo =============================================================================== 
    echo API non disponible - Suivez les instructions ci-dessus
    echo ===============================================================================
    pause
)