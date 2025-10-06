# 🪟 Guide d'Installation VRAMancer - Windows

## 🚀 Installation Rapide (Recommandée)

### Étape 1: Préparation
```bash
# 1. Ouvrir PowerShell ou Command Prompt en tant qu'administrateur
# 2. Naviguer vers le dossier VRAMancer
cd C:\path\to\VRAMancer

# 3. Vérifier Python (3.8+ requis)
python --version
```

### Étape 2: Installation Automatique
```bash
# Méthode 1: Installation complète automatique
pip install -r requirements-windows.txt

# OU Méthode 2: Installation manuelle des dépendances critiques
pip install flask flask-socketio PyQt5 requests numpy psutil torch
```

### Étape 3: Lancement
```bash
# Méthode 1: Lanceur batch (double-clic)
start_vramancer.bat

# Méthode 2: Lanceur Python intelligent
python launch_vramancer.py

# Méthode 3: Dashboard minimal de secours
python dashboard_minimal_windows.py
```

## 🔧 Résolution de Problèmes

### Problème: "Module not found"
```bash
# Diagnostic automatique et réparation
python fix_windows_dashboard.py
```

### Problème: PyQt5 ne s'installe pas
```bash
# Solution alternative
pip install PyQt5 --no-cache-dir
# OU utiliser conda
conda install pyqt
```

### Problème: Torch installation échoue
```bash
# Version CPU-only pour Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Problème: Flask-SocketIO erreurs
```bash
# Versions compatibles
pip install flask==3.0.2 flask-socketio==5.10.0 eventlet==0.35.2
```

## 📋 Tests de Validation

### Test 1: Dashboard Web
```bash
python dashboard/dashboard_web.py
# Aller à: http://localhost:5000
```

### Test 2: Dashboard Qt
```bash
python dashboard/dashboard_qt.py
# Une fenêtre Qt devrait s'ouvrir
```

### Test 3: Dashboard Tkinter
```bash
python dashboard/dashboard_tk.py
# Une fenêtre Tkinter devrait s'ouvrir
```

## 🆘 Solutions de Secours

### Si rien ne fonctionne:
1. **Dashboard Minimal**: `python dashboard_minimal_windows.py`
2. **Mode CLI**: `python cli/dashboard_cli.py`
3. **Mode Debug**: Définir `VRM_API_DEBUG=1` avant le lancement

### Configuration d'Environnement:
```bash
# Variables optionnelles
set VRM_API_DEBUG=1
set VRM_API_PORT=5030
set VRM_DASHBOARD_MINIMAL=0
```

## 🎯 Configuration Cluster Hétérogène

### Pour votre setup (EPYC + RTX3090 + MI50 + laptop + MacBook):

1. **Serveur EPYC (Nœud Maître)**:
```bash
# Sur le serveur Ubuntu sous Proxmox
python core/orchestrator/heterogeneous_manager.py --role master
```

2. **PC Portable (Nœud Worker)**:
```bash
# Sur Windows avec i5 + RTX 4060Ti
python core/orchestrator/heterogeneous_manager.py --role worker --master-ip 192.168.x.x
```

3. **MacBook M4 (Nœud Edge)**:
```bash
# Configuration Apple MPS
python core/orchestrator/heterogeneous_manager.py --role edge --backend mps
```

### Dashboard Mobile pour le Monitoring:
```bash
# Lancer le dashboard mobile sur n'importe quel nœud
python mobile/dashboard_heterogeneous.py
# Accessible depuis: http://ip-du-noeud:8080
```

## 📚 Fichiers Utiles Créés

- `fix_windows_dashboard.py` - Diagnostic et réparation automatique
- `launch_vramancer.py` - Lanceur intelligent multi-dashboard
- `start_vramancer.bat` - Lanceur batch Windows
- `dashboard_minimal_windows.py` - Dashboard de secours
- `requirements-windows.txt` - Dépendances Windows optimisées

## 🎭 Votre Idée Plug-and-Play Multi-Nœuds

Votre concept d'**auto-sensing des couches selon performance** avec **notion maître-esclave** est exactement implémenté dans:

- `core/orchestrator/heterogeneous_manager.py` - Gestion automatique des nœuds
- `mobile/dashboard_heterogeneous.py` - Supervision en temps réel
- Détection automatique des capacités (CUDA/ROCm/MPS/CPU)
- Équilibrage de charge intelligent selon les performances
- Architecture maître-esclave avec failover automatique

## 🔄 Prochaines Étapes

1. **Validation Windows**: Tester que les dashboards marchent
2. **Déploiement Multi-Nœuds**: Configuration de votre cluster hétérogène
3. **Tests de Performance**: Benchmarks CUDA + ROCm + MPS
4. **Optimisation**: Ajustement selon vos workloads spécifiques

---

💡 **Astuce**: Commencez toujours par `python fix_windows_dashboard.py` pour un diagnostic complet !