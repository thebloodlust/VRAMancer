@echo off
REM ═══════════════════════════════════════════════════════════════════
REM  VRAMancer — Windows Auto-Installer
REM  Détecte Python, GPU, installe tout automatiquement.
REM
REM  Usage:
REM    install.bat              Installation standard
REM    install.bat --full       Toutes les dépendances
REM    install.bat --lite       CLI minimal
REM    install.bat --dev        Mode développement
REM ═══════════════════════════════════════════════════════════════════

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║            VRAMancer — Windows Installer                 ║
echo  ║     Multi-GPU LLM Inference for Heterogeneous Hardware   ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

REM ── Vérifier Python ─────────────────────────────────────────────
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo  [!] Python non trouvé.
    echo.
    echo  Installer Python 3.10+ depuis:
    echo    https://www.python.org/downloads/
    echo.
    echo  Ou avec winget:
    echo    winget install Python.Python.3.12
    echo.
    pause
    exit /b 1
)

REM ── Vérifier la version Python ──────────────────────────────────
python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>nul
if %errorlevel% neq 0 (
    echo  [!] Python 3.10+ requis. Version actuelle:
    python --version
    echo.
    echo  Installer Python 3.10+ depuis:
    echo    https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "delims=" %%v in ('python --version 2^>^&1') do echo  [OK] %%v

REM ── Détecter GPU ────────────────────────────────────────────────
set GPU_TYPE=cpu
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo  [OK] GPU NVIDIA détecté
    set GPU_TYPE=nvidia
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
)

REM ── Lancer l'installeur Python ──────────────────────────────────
echo.
echo  Lancement de l'installeur universel...
echo.

python "%~dp0install.py" %*

if %errorlevel% neq 0 (
    echo.
    echo  [!] L'installation a échoué. Vérifiez les messages ci-dessus.
    pause
    exit /b 1
)

echo.
echo  Installation terminée!
echo.
echo  Pour activer l'environnement:
echo    .venv\Scripts\activate
echo.
echo  Pour lancer le serveur:
echo    vramancer-api
echo.
pause
