@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

echo ╔══════════════════════════════════════════════════════════╗
echo ║            VRAMancer — Windows Installer                 ║
echo ║     Multi-GPU LLM Inference for Heterogeneous Hardware   ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

set "INSTALL_DIR=%USERPROFILE%\VRAMancer"
set "REPO_URL=https://github.com/thebloodlust/VRAMancer.git"

:: ── Vérif Admin ──────────────────────────────────────────
echo [*] Vérification du système...

:: ── Python ───────────────────────────────────────────────
echo.
echo [*] Recherche de Python 3.10+...

set "PYTHON_CMD="
for %%P in (python3.12 python3.11 python3.10 python3 python) do (
    where %%P >nul 2>&1 && (
        for /f "tokens=*" %%V in ('%%P -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}') if v.major>=3 and v.minor>=10 else print('old')" 2^>nul') do (
            if not "%%V"=="old" (
                if not defined PYTHON_CMD (
                    set "PYTHON_CMD=%%P"
                    set "PY_VER=%%V"
                    echo   [OK] Python trouvé: %%P ^(%%V^)
                )
            )
        )
    )
)

if not defined PYTHON_CMD (
    echo   [!] Python 3.10+ non trouvé.
    echo.
    echo   Options :
    echo     1. Télécharger depuis https://www.python.org/downloads/
    echo     2. Ou via winget : winget install Python.Python.3.12
    echo.
    echo   IMPORTANT: Cochez "Add Python to PATH" pendant l'installation !
    echo.

    :: Essayer winget
    where winget >nul 2>&1
    if !errorlevel! equ 0 (
        echo   Installation automatique via winget...
        winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
        set "PYTHON_CMD=python"
        echo   [OK] Python installé. Vous devrez peut-être redémarrer ce script.
    ) else (
        echo   [ERREUR] Installez Python manuellement puis relancez ce script.
        pause
        exit /b 1
    )
)

:: ── Git ──────────────────────────────────────────────────
echo.
echo [*] Recherche de Git...

where git >nul 2>&1
if %errorlevel% neq 0 (
    echo   [!] Git non trouvé.
    where winget >nul 2>&1
    if !errorlevel! equ 0 (
        echo   Installation de Git via winget...
        winget install Git.Git --accept-package-agreements --accept-source-agreements
        echo   [OK] Git installé.
        echo   Fermez et relancez ce script pour que Git soit dans le PATH.
        pause
        exit /b 0
    ) else (
        echo   [ERREUR] Installez Git depuis https://git-scm.com/download/win
        pause
        exit /b 1
    )
) else (
    echo   [OK] Git trouvé
)

:: ── Clone / Update ───────────────────────────────────────
echo.
echo [*] VRAMancer...

:: Si lancé depuis le repo
if exist "%~dp0install.py" if exist "%~dp0pyproject.toml" (
    echo   [OK] Lancé depuis le repo VRAMancer
    set "INSTALL_DIR=%~dp0"
    goto :do_install
)

if exist "%INSTALL_DIR%\install.py" (
    echo   [OK] VRAMancer déjà présent dans %INSTALL_DIR%
    cd /d "%INSTALL_DIR%"
    echo   Mise à jour...
    git pull --ff-only 2>nul || echo   [!] git pull échoué
) else (
    echo   Clonage de VRAMancer...
    git clone "%REPO_URL%" "%INSTALL_DIR%"
)

:do_install
cd /d "%INSTALL_DIR%"
echo   Répertoire: %INSTALL_DIR%

:: ── Lancer install.py ────────────────────────────────────
echo.
echo [*] Lancement de l'installation...
echo.

%PYTHON_CMD% install.py --yes

:: ── Résumé ───────────────────────────────────────────────
echo.
echo ═══════════════════════════════════════════════════════
echo   Installation terminée !
echo ═══════════════════════════════════════════════════════
echo.
echo   Dossier : %INSTALL_DIR%
echo.
echo   Démarrage :
echo     %INSTALL_DIR%\.venv\Scripts\activate
echo     vramancer serve
echo.
echo   API : http://localhost:5030
echo.

pause
