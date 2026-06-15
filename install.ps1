# VRAMancer — installateur une-ligne Windows (PowerShell).
#
#   irm https://raw.githubusercontent.com/thebloodlust/VRAMancer/main/install.ps1 | iex
#
# Miroir de install.sh : vérifie les prérequis, récupère le repo, délègue au
# install.py éprouvé (détecte GPU/CUDA, venv isolé, bon wheel torch), puis pose
# la commande `vramancer`. Idempotent.
#
# Variables d'env : VRAMANCER_HOME, VRAMANCER_MODE (standard|full|lite|dev),
#                   VRAMANCER_REPO, VRAMANCER_DRY_RUN (1 = ne lance pas install.py)

$ErrorActionPreference = "Stop"

$Repo    = if ($env:VRAMANCER_REPO) { $env:VRAMANCER_REPO } else { "https://github.com/thebloodlust/VRAMancer.git" }
$HomeDir = if ($env:VRAMANCER_HOME) { $env:VRAMANCER_HOME } else { "$env:USERPROFILE\.vramancer" }
$Mode    = if ($env:VRAMANCER_MODE) { $env:VRAMANCER_MODE } else { "standard" }
$BinDir  = "$env:LOCALAPPDATA\VRAMancer\bin"
$Dry     = if ($env:VRAMANCER_DRY_RUN) { $env:VRAMANCER_DRY_RUN } else { "0" }

function Step($m) { Write-Host "`n> $m" -ForegroundColor Cyan }
function Info($m) { Write-Host "  [OK] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  [!] $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "  [x] $m" -ForegroundColor Red; exit 1 }

Step "VRAMancer — installateur une-ligne (Windows)"

# 1) Python (py launcher ou python), >= 3.10
$PyExe = $null; $PyArg = @()
if (Get-Command py -ErrorAction SilentlyContinue)          { $PyExe = "py"; $PyArg = @("-3") }
elseif (Get-Command python -ErrorAction SilentlyContinue)  { $PyExe = "python" }
elseif (Get-Command python3 -ErrorAction SilentlyContinue) { $PyExe = "python3" }
if (-not $PyExe) { Die "Python introuvable (>= 3.10 requis). https://www.python.org/downloads/" }

$verRaw = (& $PyExe @PyArg --version 2>&1) | Out-String         # "Python 3.12.3"
$ver = ($verRaw -replace '[^0-9.]', '').Trim('.')
$parts = $ver.Split('.'); $maj = [int]$parts[0]; $min = [int]$parts[1]
if ($maj -lt 3 -or ($maj -eq 3 -and $min -lt 10)) { Die "Python $ver trop ancien (>= 3.10)." }
Info "Python $ver ($PyExe $($PyArg -join ' '))"

# Driver NVIDIA (facultatif — install.py choisit le bon wheel)
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $drv = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1)
    Info "Driver NVIDIA: $drv"
} else {
    Warn "nvidia-smi absent — install CPU (install.py choisira le wheel)."
}

# 2) Sources : checkout local si lancé depuis le repo, sinon clone/pull
$SelfDir = if ($PSScriptRoot) { $PSScriptRoot } elseif ($MyInvocation.MyCommand.Path) { Split-Path -Parent $MyInvocation.MyCommand.Path } else { $null }
if ($SelfDir -and (Test-Path (Join-Path $SelfDir "install.py"))) {
    $Src = $SelfDir
    Info "Sources locales détectées: $Src"
} else {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Die "git introuvable (nécessaire pour cloner)." }
    if (Test-Path (Join-Path $HomeDir ".git")) {
        Step "Mise à jour du dépôt ($HomeDir)"; & git -C $HomeDir pull --ff-only --quiet; Info "À jour"
    } else {
        Step "Clonage de $Repo"; & git clone --depth 1 $Repo $HomeDir --quiet; Info "Cloné dans $HomeDir"
    }
    $Src = $HomeDir
}

# 3) Déléguer au Universal Auto-Installer
Step "Installation (install.py, mode=$Mode)"
$ModeFlag = switch ($Mode) { "full" { "--full" } "lite" { "--lite" } "dev" { "--dev" } default { "" } }
if ($Dry -eq "1") {
    Warn "DRY_RUN: install.py non lancé (test du bootstrap)."
} else {
    Push-Location $Src
    $pyArgs = @($PyArg) + @("install.py"); if ($ModeFlag) { $pyArgs += $ModeFlag }
    & $PyExe @pyArgs
    Pop-Location
}

# 4) Poser la commande `vramancer`
Step "Commande vramancer"
$VenvPy = Join-Path $Src ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
$Shim = Join-Path $BinDir "vramancer.cmd"
"@echo off`r`n`"$VenvPy`" -m vramancer %*" | Set-Content -Path $Shim -Encoding ASCII
Info "Installée: $Shim"

# 5) PATH utilisateur + étapes suivantes
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$BinDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$BinDir", "User")
    Warn "$BinDir ajouté au PATH utilisateur — rouvre ton terminal pour qu'il prenne effet."
}

Step "Terminé"
Write-Host "  Essaie :  vramancer quickstart code-assistant" -ForegroundColor Cyan
Write-Host "  Puis   :  vramancer quickstart code-assistant --run" -ForegroundColor Cyan
