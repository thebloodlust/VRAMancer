#!/usr/bin/env python3
"""
VRAMancer — Universal Auto-Installer
=====================================
Détecte automatiquement l'OS, le GPU et la version Python,
puis installe VRAMancer avec le bon backend PyTorch.

Usage:
    python install.py              # Installation standard
    python install.py --full       # Toutes les dépendances (GUI, tracing, etc.)
    python install.py --lite       # CLI minimal
    python install.py --docker     # Build + lance le stack Docker Compose
    python install.py --dev        # Mode développement (tests, lint, etc.)
    python install.py --no-venv    # Installe dans l'environnement courant
    python install.py --service    # Configure un service systemd/launchd
"""

import os
import sys
import shutil
import platform
import subprocess
import argparse
import json
import secrets
import textwrap
from pathlib import Path

# ── Constantes ────────────────────────────────────────────────────────
MIN_PYTHON = (3, 10)
VENV_DIR = ".venv"
PYTORCH_INDEX = {
    "cuda_12": "https://download.pytorch.org/whl/cu121",
    "cuda_11": "https://download.pytorch.org/whl/cu118",
    "rocm":    "https://download.pytorch.org/whl/rocm6.0",
    "cpu":     "https://download.pytorch.org/whl/cpu",
}
BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║            VRAMancer — Universal Installer               ║
║     Multi-GPU LLM Inference for Heterogeneous Hardware   ║
╚══════════════════════════════════════════════════════════╝
"""

# ── Couleurs (désactivées si pas de TTY) ──────────────────────────────
NO_COLOR = not sys.stdout.isatty() or os.environ.get("NO_COLOR")

def _c(code: str, text: str) -> str:
    return text if NO_COLOR else f"\033[{code}m{text}\033[0m"

def green(t: str)  -> str: return _c("32", t)
def yellow(t: str) -> str: return _c("33", t)
def red(t: str)    -> str: return _c("31", t)
def cyan(t: str)   -> str: return _c("36", t)
def bold(t: str)   -> str: return _c("1", t)

def info(msg: str):  print(f"  {green('✓')} {msg}")
def warn(msg: str):  print(f"  {yellow('⚠')} {msg}")
def fail(msg: str):  print(f"  {red('✗')} {msg}")
def step(msg: str):  print(f"\n{bold(cyan(f'▸ {msg}'))}")


# ── Détection OS ──────────────────────────────────────────────────────
def detect_os() -> dict:
    """Retourne un dict décrivant l'OS courant."""
    s = platform.system().lower()
    info_dict = {
        "system": s,
        "release": platform.release(),
        "machine": platform.machine(),
        "is_wsl": False,
    }
    if s == "linux":
        # Vérifier WSL
        try:
            with open("/proc/version", "r") as f:
                if "microsoft" in f.read().lower():
                    info_dict["is_wsl"] = True
        except FileNotFoundError:
            pass
        # Distro
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        info_dict["distro"] = line.split("=", 1)[1].strip().strip('"')
                        break
        except FileNotFoundError:
            info_dict["distro"] = "Linux (unknown)"
    elif s == "darwin":
        ver = platform.mac_ver()[0]
        info_dict["macos_version"] = ver
    return info_dict


# ── Détection GPU ─────────────────────────────────────────────────────
def detect_gpu() -> dict:
    """Détecte les GPUs disponibles et retourne le backend recommandé."""
    result = {"backend": "cpu", "gpus": [], "cuda_version": None}

    # NVIDIA
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if out.returncode == 0:
                for line in out.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        result["gpus"].append({
                            "name": parts[0],
                            "vram_mb": int(parts[1]),
                            "driver": parts[2],
                        })
                # Déterminer version CUDA du driver
                try:
                    out2 = subprocess.run(
                        ["nvidia-smi", "--query-gpu=driver_version",
                         "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=5
                    )
                    driver_ver = out2.stdout.strip().split("\n")[0].strip()
                    major = int(driver_ver.split(".")[0])
                    # Driver >= 525 supporte CUDA 12, sinon CUDA 11
                    if major >= 525:
                        result["backend"] = "cuda_12"
                        result["cuda_version"] = "12.1"
                    else:
                        result["backend"] = "cuda_11"
                        result["cuda_version"] = "11.8"
                except Exception:
                    result["backend"] = "cuda_12"
                    result["cuda_version"] = "12.1"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # AMD ROCm
    if result["backend"] == "cpu":
        rocm_smi = shutil.which("rocm-smi") or shutil.which("rocminfo")
        if rocm_smi:
            try:
                out = subprocess.run(["rocm-smi", "--showproductname"],
                                     capture_output=True, text=True, timeout=10)
                if out.returncode == 0:
                    result["backend"] = "rocm"
                    result["gpus"].append({"name": "AMD ROCm GPU", "vram_mb": 0})
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    # Apple Silicon
    if result["backend"] == "cpu" and platform.system() == "Darwin":
        machine = platform.machine()
        if machine == "arm64":
            result["backend"] = "mps"
            # Estimer VRAM (mémoire unifiée)
            try:
                out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                     capture_output=True, text=True, timeout=5)
                mem_bytes = int(out.stdout.strip())
                result["gpus"].append({
                    "name": f"Apple Silicon ({machine})",
                    "vram_mb": mem_bytes // (1024 * 1024),
                })
            except Exception:
                result["gpus"].append({"name": f"Apple Silicon ({machine})", "vram_mb": 0})

    return result


# ── Vérification Python ──────────────────────────────────────────────
def check_python() -> bool:
    v = sys.version_info
    if (v.major, v.minor) < MIN_PYTHON:
        fail(f"Python {v.major}.{v.minor} détecté — Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ requis")
        return False
    info(f"Python {v.major}.{v.minor}.{v.micro}")
    return True


# ── Création du venv ──────────────────────────────────────────────────
def create_venv(base_dir: Path) -> Path:
    venv_path = base_dir / VENV_DIR
    if venv_path.exists():
        info(f"Virtualenv existant: {venv_path}")
    else:
        info(f"Création du virtualenv: {venv_path}")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
    # Retourner le chemin du pip dans le venv
    if platform.system() == "Windows":
        return venv_path / "Scripts"
    return venv_path / "bin"


def get_pip(bin_dir: Path) -> str:
    pip = bin_dir / ("pip.exe" if platform.system() == "Windows" else "pip")
    return str(pip)


def get_python(bin_dir: Path) -> str:
    py = bin_dir / ("python.exe" if platform.system() == "Windows" else "python")
    return str(py)


# ── Installation PyTorch ──────────────────────────────────────────────
def install_pytorch(pip_cmd: str, gpu_info: dict):
    backend = gpu_info["backend"]
    step("Installation de PyTorch")

    if backend == "mps":
        # macOS ARM — PyTorch standard inclut MPS
        info("Apple Silicon détecté → PyTorch avec support MPS")
        subprocess.check_call([pip_cmd, "install", "--upgrade",
                               "torch", "torchvision", "torchaudio"])
    elif backend in ("cuda_12", "cuda_11"):
        index_url = PYTORCH_INDEX[backend]
        cuda_ver = gpu_info.get("cuda_version", "12.1")
        info(f"NVIDIA GPU détecté → PyTorch CUDA {cuda_ver}")
        subprocess.check_call([pip_cmd, "install", "--upgrade",
                               "torch", "torchvision", "torchaudio",
                               "--index-url", index_url])
    elif backend == "rocm":
        index_url = PYTORCH_INDEX["rocm"]
        info("AMD ROCm détecté → PyTorch ROCm")
        subprocess.check_call([pip_cmd, "install", "--upgrade",
                               "torch", "torchvision", "torchaudio",
                               "--index-url", index_url])
    else:
        info("Pas de GPU détecté → PyTorch CPU")
        index_url = PYTORCH_INDEX["cpu"]
        subprocess.check_call([pip_cmd, "install", "--upgrade",
                               "torch", "torchvision", "torchaudio",
                               "--index-url", index_url])


# ── Installation VRAMancer ────────────────────────────────────────────
def install_vramancer(pip_cmd: str, mode: str, base_dir: Path):
    step("Installation de VRAMancer")

    extras = {
        "full":  ".[all]",
        "lite":  ".[lite]",
        "dev":   ".[test,dev]",
        "server": ".[server,security,compression]",
    }
    target = extras.get(mode, ".")

    info(f"Mode: {bold(mode)} → pip install -e {target}")
    subprocess.check_call([pip_cmd, "install", "--upgrade", "pip", "setuptools", "wheel"])
    subprocess.check_call([pip_cmd, "install", "-e", target], cwd=str(base_dir))


# ── Génération du token API ──────────────────────────────────────────
def generate_env_file(base_dir: Path):
    env_file = base_dir / ".env"
    if env_file.exists():
        info(".env existant — pas de modification")
        return

    token = secrets.token_urlsafe(32)
    env_file.write_text(textwrap.dedent(f"""\
        # VRAMancer — Variables d'environnement
        # Généré automatiquement par install.py
        VRM_API_TOKEN={token}
        # VRM_LOG_JSON=1
        # VRM_TRACING=1
        # VRM_SQLITE_PATH=./vramancer.db
    """))
    info(f"Fichier .env créé avec token API: {token[:8]}…")


# ── Docker Compose ────────────────────────────────────────────────────
def setup_docker(base_dir: Path):
    step("Docker Compose")
    docker = shutil.which("docker")
    if not docker:
        fail("Docker non trouvé. Installe Docker: https://docs.docker.com/get-docker/")
        return False

    compose_file = base_dir / "docker-compose.yml"
    if not compose_file.exists():
        fail("docker-compose.yml introuvable")
        return False

    info("Construction des images Docker…")
    subprocess.check_call(["docker", "compose", "build"], cwd=str(base_dir))
    info("Lancement du stack…")
    subprocess.check_call(["docker", "compose", "up", "-d"], cwd=str(base_dir))
    info("Stack VRAMancer lancé!")
    info("  API:         http://localhost:5030")
    info("  Prometheus:  http://localhost:9090")
    info("  Grafana:     http://localhost:3000")
    info("  Alertmanager:http://localhost:9093")
    return True


# ── Service systemd (Linux) ──────────────────────────────────────────
def install_systemd_service(base_dir: Path, bin_dir: Path):
    step("Configuration du service systemd")

    if platform.system() != "Linux":
        warn("Les services systemd ne sont disponibles que sous Linux")
        return

    python_bin = get_python(bin_dir)
    service_content = textwrap.dedent(f"""\
        [Unit]
        Description=VRAMancer Multi-GPU LLM Inference Server
        After=network.target
        Wants=network-online.target

        [Service]
        Type=simple
        User={os.environ.get('USER', 'vramancer')}
        WorkingDirectory={base_dir}
        EnvironmentFile={base_dir}/.env
        ExecStart={python_bin} -m gunicorn 'core.production_api:create_app()' \\
            --bind 0.0.0.0:5030 --workers 2 --timeout 300
        Restart=always
        RestartSec=5
        LimitNOFILE=65536

        [Install]
        WantedBy=multi-user.target
    """)

    service_path = Path("/etc/systemd/system/vramancer.service")
    try:
        service_path.write_text(service_content)
        subprocess.check_call(["systemctl", "daemon-reload"])
        subprocess.check_call(["systemctl", "enable", "vramancer"])
        info("Service systemd installé et activé")
        info("  Démarrer:  sudo systemctl start vramancer")
        info("  Logs:      journalctl -u vramancer -f")
    except PermissionError:
        user_service_dir = Path.home() / ".config" / "systemd" / "user"
        user_service_dir.mkdir(parents=True, exist_ok=True)
        user_path = user_service_dir / "vramancer.service"
        user_path.write_text(service_content)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        subprocess.run(["systemctl", "--user", "enable", "vramancer"], check=False)
        info(f"Service utilisateur installé: {user_path}")
        info("  Démarrer:  systemctl --user start vramancer")


# ── Smoke test ────────────────────────────────────────────────────────
def run_smoke_test(python_cmd: str, base_dir: Path) -> bool:
    step("Vérification de l'installation")

    # Test 1: import
    try:
        result = subprocess.run(
            [python_cmd, "-c",
             "import core; print(f'VRAMancer v{core.__version__}')"],
            capture_output=True, text=True, timeout=30, cwd=str(base_dir)
        )
        if result.returncode == 0:
            info(result.stdout.strip())
        else:
            fail(f"Import échoué: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        fail("Timeout à l'import")
        return False

    # Test 2: CLI help
    try:
        result = subprocess.run(
            [python_cmd, "-m", "vramancer.main", "--help"],
            capture_output=True, text=True, timeout=10, cwd=str(base_dir)
        )
        if result.returncode == 0:
            info("CLI vramancer OK")
        else:
            warn("CLI help a retourné un code non-zero (non fatal)")
    except Exception:
        warn("CLI help indisponible (non fatal)")

    # Test 3: PyTorch
    try:
        result = subprocess.run(
            [python_cmd, "-c", textwrap.dedent("""\
                import torch
                print(f'PyTorch {torch.__version__}')
                if torch.cuda.is_available():
                    n = torch.cuda.device_count()
                    for i in range(n):
                        name = torch.cuda.get_device_name(i)
                        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
                        print(f'  GPU {i}: {name} ({mem:.1f} GB)')
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    print('  Apple MPS disponible')
                else:
                    print('  Mode CPU uniquement')
            """)],
            capture_output=True, text=True, timeout=30, cwd=str(base_dir)
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                info(line)
        else:
            warn(f"PyTorch non disponible: {result.stderr.strip()[:100]}")
    except Exception:
        warn("Test PyTorch échoué (non fatal)")

    return True


# ── Résumé final ──────────────────────────────────────────────────────
def print_summary(os_info: dict, gpu_info: dict, mode: str, venv_used: bool):
    step("Installation terminée!")
    print()
    activate = ""
    if venv_used:
        if os_info["system"] == "windows":
            activate = f"  {cyan('.venv\\Scripts\\activate')}"
        else:
            activate = f"  {cyan('source .venv/bin/activate')}"

    gpu_names = ", ".join(g["name"] for g in gpu_info["gpus"]) if gpu_info["gpus"] else "CPU only"

    print(f"""
  {bold('Système')}:   {os_info.get('distro', os_info.get('macos_version', os_info['system']))}
  {bold('GPU')}:       {gpu_names}
  {bold('Backend')}:   {gpu_info['backend']}
  {bold('Mode')}:      {mode}
""")

    if activate:
        print(f"  {bold('Activer le venv')}:")
        print(f"  {activate}")
        print()

    print(f"  {bold('Démarrage rapide')}:")
    prefix = "  " if not venv_used else "  "
    print(f"{prefix}{cyan('vramancer-api')}                        # Lancer le serveur")
    print(f"{prefix}{cyan('curl http://localhost:5030/health')}    # Vérifier la santé")
    print(f"{prefix}{cyan('vramancer --help')}                    # Aide CLI")
    print()
    print(f"  {bold('Documentation')}: https://github.com/votre-org/VRAMancer#readme")
    print()


# ── Point d'entrée ───────────────────────────────────────────────────
def main():
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="VRAMancer — Installeur automatique multi-plateformes"
    )
    parser.add_argument("--full", action="store_true",
                        help="Installer toutes les dépendances (GUI, tracing, etc.)")
    parser.add_argument("--lite", action="store_true",
                        help="Installation minimale (CLI uniquement)")
    parser.add_argument("--dev", action="store_true",
                        help="Mode développement (tests, lint, etc.)")
    parser.add_argument("--server", action="store_true",
                        help="Mode serveur (compression, sécurité, tracing)")
    parser.add_argument("--docker", action="store_true",
                        help="Build et lance le stack Docker Compose")
    parser.add_argument("--no-venv", action="store_true",
                        help="Ne pas créer de virtualenv")
    parser.add_argument("--no-pytorch", action="store_true",
                        help="Ne pas installer PyTorch (si déjà installé)")
    parser.add_argument("--service", action="store_true",
                        help="Configurer en service systemd/launchd")
    parser.add_argument("--skip-test", action="store_true",
                        help="Ne pas lancer le smoke test")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Accepter toutes les confirmations")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    # ── Étape 1: Vérifications ──
    step("Détection de l'environnement")

    if not check_python():
        sys.exit(1)

    os_info = detect_os()
    info(f"OS: {os_info.get('distro', os_info.get('macos_version', os_info['system']))}"
         + (" (WSL)" if os_info.get("is_wsl") else ""))

    gpu_info = detect_gpu()
    if gpu_info["gpus"]:
        for g in gpu_info["gpus"]:
            vram = f" ({g['vram_mb']} MB)" if g.get("vram_mb") else ""
            info(f"GPU: {g['name']}{vram}")
    else:
        warn("Aucun GPU détecté — mode CPU")

    info(f"Backend PyTorch recommandé: {bold(gpu_info['backend'])}")

    # ── Docker shortcut ──
    if args.docker:
        success = setup_docker(base_dir)
        sys.exit(0 if success else 1)

    # ── Étape 2: Mode d'installation ──
    if args.full:
        mode = "full"
    elif args.lite:
        mode = "lite"
    elif args.dev:
        mode = "dev"
    elif args.server:
        mode = "server"
    else:
        mode = "standard"

    # ── Étape 3: Virtualenv ──
    if args.no_venv:
        bin_dir = Path(os.path.dirname(sys.executable))
        pip_cmd = shutil.which("pip") or shutil.which("pip3") or "pip"
        python_cmd = sys.executable
        venv_used = False
    else:
        step("Configuration du virtualenv")
        bin_dir = create_venv(base_dir)
        pip_cmd = get_pip(bin_dir)
        python_cmd = get_python(bin_dir)
        venv_used = True

    # ── Étape 4: PyTorch ──
    if not args.no_pytorch:
        install_pytorch(pip_cmd, gpu_info)
    else:
        info("Installation PyTorch ignorée (--no-pytorch)")

    # ── Étape 5: VRAMancer ──
    install_vramancer(pip_cmd, mode, base_dir)

    # ── Étape 6: Fichier .env ──
    step("Configuration")
    generate_env_file(base_dir)

    # ── Étape 7: Service systemd ──
    if args.service:
        install_systemd_service(base_dir, bin_dir)

    # ── Étape 8: Smoke test ──
    if not args.skip_test:
        run_smoke_test(python_cmd, base_dir)

    # ── Résumé ──
    print_summary(os_info, gpu_info, mode, venv_used)


if __name__ == "__main__":
    main()
