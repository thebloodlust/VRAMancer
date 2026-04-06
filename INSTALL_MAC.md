# Installation VRAMancer MLX Worker — MacBook M4

## Étape 1 : Créer un venv arm64 et installer les dépendances

**IMPORTANT** : MLX nécessite un Python **arm64 natif** (pas Rosetta/x86).

Même si tu installes Python 3.14 arm64 depuis python.org, le `python3` dans le PATH peut encore pointer vers une version x86 (Homebrew x86, ancienne install, etc.).

**Vérifier l'architecture :**
```bash
python3 -c "import platform; print(platform.machine())"
```
→ Doit afficher `arm64`. Si ça affiche `x86_64`, il faut utiliser le **chemin complet** du Python arm64.

**Trouver le bon Python arm64 :**
```bash
# Python installé depuis python.org (3.14)
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 -c "import platform; print(platform.machine())"

# Python Homebrew arm64 (si installé via /opt/homebrew)
/opt/homebrew/bin/python3 -c "import platform; print(platform.machine())"

# Chercher tous les Python disponibles
find /Library/Frameworks /opt/homebrew/bin /usr/local/bin -name "python3*" 2>/dev/null
```

**Créer le venv avec le bon Python (celui qui affiche arm64) :**
```bash
# Exemple avec Python 3.14 installé depuis python.org :
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 -m venv ~/venv_vrm

# OU avec Homebrew arm64 :
# /opt/homebrew/bin/python3 -m venv ~/venv_vrm

# Activer et vérifier que c'est bien arm64
source ~/venv_vrm/bin/activate
python3 -c "import platform; print(platform.machine())"
# → DOIT afficher arm64

# Installer les dépendances
pip install mlx mlx-lm numpy
```

### Si les deux Python (x64 et arm64) cohabitent dans le même dossier

macOS peut avoir un binaire universel (fat binary) qui contient les deux architectures. Par défaut il peut choisir x86_64 si le Terminal tourne sous Rosetta.

**ATTENTION : un venv "bake" l'architecture du Python qui l'a créé.** Si tu as créé `~/venv_vrm` quand le terminal était en x86, le venv restera x86 même après `source`. **Il faut le supprimer et le recréer :**

```bash
# 1. Supprimer l'ancien venv x86
deactivate 2>/dev/null
rm -rf ~/venv_vrm

# 2. Forcer arm64 et recréer
arch -arm64 python3 -c "import platform; print(platform.machine())"
# → doit afficher arm64

arch -arm64 python3 -m venv ~/venv_vrm
source ~/venv_vrm/bin/activate

# 3. Vérifier que le venv est bien arm64
python3 -c "import platform; print(platform.machine())"
# → DOIT afficher arm64

# 4. Installer
pip install mlx mlx-lm numpy
```

**Si `arch -arm64` ne suffit pas**, utilise le chemin complet :
```bash
arch -arm64 /Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 -m venv ~/venv_vrm
```

**Vérifier que le Terminal lui-même n'est pas en Rosetta :**
1. Finder → Applications → Utilitaires → Terminal
2. Clic droit → Lire les informations
3. **Décocher** "Ouvrir avec Rosetta" si c'est coché
4. Fermer et rouvrir le Terminal

**Désinstaller le Python x86 (optionnel) :**
```bash
# Si c'est un Homebrew x86 (installé sous /usr/local)
arch -x86_64 /usr/local/bin/brew uninstall python3

# Si c'est un installeur python.org x86 (adapter la version)
sudo rm -rf /Library/Frameworks/Python.framework/Versions/3.12
sudo rm -f /usr/local/bin/python3.12 /usr/local/bin/pip3.12
sudo rm -rf "/Applications/Python 3.12"
```

**Après désinstallation / fix, vérifier :**
```bash
source ~/venv_vrm/bin/activate
python3 -c "import platform; print(platform.machine())"
# → DOIT afficher arm64, sinon MLX ne s'installera pas
pip install mlx mlx-lm numpy
```

## Étape 2 : Télécharger le worker

```bash
curl -L -o ~/mac_worker.py https://raw.githubusercontent.com/thebloodlust/VRAMancer/main/mac_worker.py
```

## Étape 3 : Lancer le worker

```bash
source ~/venv_vrm/bin/activate
python3 ~/mac_worker.py --model mlx-community/Qwen2.5-14B-4bit --start-layer 42 --end-layer 48
```

Le modèle sera téléchargé automatiquement (~8 GB, une seule fois).

Quand tu vois :
```
Freed 42 unused layers + embed/norm/head

==================================================
  VTP MLX Compute Worker
  Model:   mlx-community/Qwen2.5-14B-4bit
  Layers:  42-47 (6 layers)
  Listen:  0.0.0.0:18951
  Backend: Apple Silicon MLX (Metal)
==================================================
Waiting for connections...
```

C'est prêt. Lance le benchmark depuis Ubuntu.

## Étape 4 : Benchmark (côté Ubuntu)

```bash
cd /home/jeremie/VRAMancer/VRAMancer
source .venv/bin/activate
python benchmarks/bench_3node.py --mac-host 192.168.1.27
```

## Résumé

| Machine | Rôle | Layers |
|---|---|---|
| Ubuntu GPU0 (RTX 3090) | Layers 0-29 + embed/norm/head | 30 |
| Ubuntu GPU1 (RTX 5070 Ti) | Layers 30-41 | 12 |
| MacBook M4 (MLX Metal) | Layers 42-47 (4-bit) | 6 |
