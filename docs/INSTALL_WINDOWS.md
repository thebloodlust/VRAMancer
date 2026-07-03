# Installation VRAMancer — Windows

## Installation Rapide Infaillible (Nouveau)

Fini les bugs avec Python et PowerShell ! Utilisez notre installateur 100% autonome.

1. **Cliquez ici pour obtenir l'installateur :** 
   👉 [Télécharger Installer_Windows.bat](https://raw.githubusercontent.com/thebloodlust/VRAMancer/main/scripts/Installer_Windows.bat)
2. Faites un **Clic-droit sur la page > "Enregistrer sous..."**
3. Sauvegardez le fichier `Installer_Windows.bat` sur votre **Bureau**.
4. Double-cliquez sur `Installer_Windows.bat`.

Il va automatiquement télécharger un Python portable, les paquets VRAMancer, PyTorch CUDA et créer un raccourci de lancement sur votre bureau !

---

## Installation Manuelle (Avancée)

Si vous préférez contrôler l'environnement vous-même :

### 1. Pré-requis

- **Python 3.10+** → https://python.org/downloads (cocher "Add to PATH")
- **Git** (optionnel) → https://git-scm.com/download/win

### 2. Télécharger et installer

```powershell
# Ouvrir PowerShell, puis :
git clone https://github.com/thebloodlust/VRAMancer.git
cd VRAMancer

# Créer un environnement isolé
python -m venv .venv
.venv\Scripts\Activate.ps1

# Installer
pip install -e .
```

### 3. Installer PyTorch pour votre GPU

```powershell
# NVIDIA (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# AMD (ROCm) — pas encore supporté sur Windows, utiliser CPU
# CPU uniquement
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Vérifier

```powershell
python -c "import core; print(f'VRAMancer v{core.__version__} OK')"

# Tests (aucun GPU nécessaire)
$env:VRM_MINIMAL_TEST="1"
$env:VRM_DISABLE_RATE_LIMIT="1"
$env:VRM_TEST_MODE="1"
pytest tests/ -q --no-cov
```

### 5. Lancer

```powershell
$env:VRM_API_TOKEN="mon-token-secret"
python -m vramancer.main --api
```

Serveur sur http://localhost:5000. Testez :

```powershell
# Charger GPT-2
curl -X POST http://localhost:5000/api/models/load `
  -H "Content-Type: application/json" `
  -H "X-API-TOKEN: mon-token-secret" `
  -d '{"model": "gpt2", "num_gpus": 1}'

# Générer
curl http://localhost:5000/v1/completions `
  -H "Content-Type: application/json" `
  -H "X-API-TOKEN: mon-token-secret" `
  -d '{"prompt": "Hello world", "max_tokens": 50}'
```

---

## Docker Desktop (alternative)

Si vous avez Docker Desktop avec WSL2 + GPU support :

```powershell
$env:VRM_API_TOKEN="mon-token-secret"
docker compose up -d
```

| Service | URL |
|---------|-----|
| API | http://localhost:5030 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

---

## Dépannage Windows

| Problème | Solution |
|----------|----------|
| `pip` non reconnu | Réinstaller Python en cochant "Add to PATH" |
| `tokenizers` échoue | `pip install tokenizers --no-build-isolation` ou installer Rust : https://rustup.rs |
| `.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `torch.cuda.is_available()` = False | Installer les drivers NVIDIA + CUDA Toolkit 12.x |
| Port 5000 occupé | `$env:VRM_API_PORT="5050"` avant de lancer |

---

## Désinstallation

```powershell
pip uninstall vramancer
Remove-Item -Recurse .venv
```

1. **Validation Windows**: Tester que les dashboards marchent
2. **Déploiement Multi-Nœuds**: Configuration de votre cluster hétérogène
3. **Tests de Performance**: Benchmarks CUDA + ROCm + MPS
4. **Optimisation**: Ajustement selon vos workloads spécifiques

---

💡 **Astuce**: Commencez toujours par `python fix_windows_dashboard.py` pour un diagnostic complet !