@echo off
setlocal enabledelayedexpansion
title VRAMancer Swarm Node Installer
color 0B

echo.
echo    __      __  _____       _      __  __                                 
echo    \ \    / / ^|  __ \     / \    ^|  \/  ^|                                
echo     \ \  / /  ^| ^|__) ^|   / _ \   ^| \  / ^|  __ _   _ __     ___    ___   _ __ 
echo      \ \/ /   ^|  _  /   / ___ \  ^| ^|\/^| ^| / _` ^| ^| '_ \   / __^|  / _ \ ^| '__^|
echo       \  /    ^| ^| \ \  / ___ \ ^| ^|  ^| ^| ^| (_^| ^| ^| ^| ^| ^| ^| (__  ^|  __/ ^| ^|   
echo        \/     ^|_^|  \_\/_/   \_\^|_^|  ^|_^|  \__,_^| ^|_^| ^|_^|  \___^|  \___^| ^|_^|   
echo.
echo ==============================================================================
echo        Bienvenue dans l'Installeur VRAMancer Node (Zero-Config)
echo ==============================================================================
echo.
echo [1] Verification de l'environnement Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe ou n'est pas dans le PATH.
    echo Veuillez installer Python 3.10+ (cochez "Add to PATH" lors de l'installation).
    pause
    exit /b
)

echo [2] Creation de l'environnement virtuel isole (Mycelium Env)...
if not exist "venv_vramancer" (
    python -m venv venv_vramancer
    echo Environnement cree avec succes !
) else (
    echo L'environnement existe deja.
)

echo [3] Activation de l'environnement et installation des dependances (Patientez...)...
call venv_vramancer\Scripts\activate.bat

echo Installation du coeur VRAMancer et du routage Rust (Zero-Copy)...
pip install --upgrade pip >nul
pip install -r requirements-windows.txt >nul 2>&1
pip install rich >nul

echo.
echo ==============================================================================
echo                       INSTALLATION TERMINEE AVEC SUCCES
echo ==============================================================================
echo.
echo Generation de votre cle P2P Swarm unique...
python -c "from vramancer.cli.swarm_cli import ui_auth_generate; ui_auth_generate()"

echo.
echo [4] Demarrage du noeud VRAMancer en arriere-plan...
echo Le noeud ecoute desormais les paquets P2P de vos amis !
echo Laissez cette fenetre ouverte pour fournir votre puissance.
echo.
python vramancer/main.py serve --backend auto
pause
