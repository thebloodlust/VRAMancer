#!/bin/bash
set -e

echo "=== Installation de VRAMancer ==="
echo ""

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux*)   PLATFORM=Linux ;;
    Darwin*)  PLATFORM=macOS ;;
    *)        PLATFORM=Unknown ;;
esac
echo "Plateforme détectée : $PLATFORM"

# Check Python
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        VERSION=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        MAJOR=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        MINOR=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERREUR : Python 3.10+ requis. Installez-le :"
    echo "  Ubuntu/Debian : sudo apt install python3 python3-venv python3-pip"
    echo "  macOS         : brew install python@3.12"
    exit 1
fi
echo "Python trouvé : $PYTHON ($VERSION)"

# Create venv
if [ ! -d ".venv" ]; then
    echo ""
    echo "Création de l'environnement virtuel..."
    $PYTHON -m venv .venv
fi

source .venv/bin/activate

# Upgrade pip
echo ""
echo "Mise à jour de pip..."
pip install --upgrade pip setuptools wheel -q

# Install VRAMancer
echo ""
echo "Installation de VRAMancer..."
pip install -e . -q

# Detect GPU and suggest PyTorch
echo ""
echo "=== Détection GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU NVIDIA détecté :"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
    echo "Pour activer CUDA (si pas encore fait) :"
    echo "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
elif [ -d "/opt/rocm" ] || command -v rocm-smi >/dev/null 2>&1; then
    echo "GPU AMD (ROCm) détecté."
    echo "Pour activer ROCm :"
    echo "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2"
elif [ "$PLATFORM" = "macOS" ] && [ "$(uname -m)" = "arm64" ]; then
    echo "Apple Silicon détecté — MPS sera utilisé automatiquement."
    echo "  pip install torch"
else
    echo "Aucun GPU détecté. VRAMancer fonctionnera en mode CPU."
    echo "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
fi

# Verify
echo ""
echo "=== Vérification ==="
python -c "import core; print(f'VRAMancer v{core.__version__} installé avec succès')"

# Check torch
python -c "
try:
    import torch
    gpu = 'CUDA' if torch.cuda.is_available() else ('MPS' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'CPU')
    print(f'PyTorch {torch.__version__} — backend : {gpu}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / 1e9
            print(f'  GPU {i}: {name} ({mem:.1f} GB)')
except ImportError:
    print('PyTorch non installé — installez-le pour utiliser vos GPUs (voir ci-dessus)')
"

echo ""
echo "=== Installation terminée ==="
echo ""
echo "Prochaines étapes :"
echo "  source .venv/bin/activate"
echo "  export VRM_API_TOKEN=mon-token-secret"
echo "  python -m vramancer.main --api"
echo ""
echo "Documentation : docs/INSTALL_ULTRA_DEBUTANT.md"
