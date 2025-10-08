# 🚀 VRAMancer - Guide de Lancement RTX 4060

## ⚡ Méthode Recommandée : System Tray

**Un seul fichier à lancer pour tout :**

```bash
systray_vramancer.bat
```

### 🎯 Pourquoi le System Tray ?

- **🖱️ Menu complet** : Clic droit sur l'icône → Accès à toutes les interfaces
- **📍 Toujours visible** : Icône VRAMancer dans la barre des tâches Windows
- **🚀 Lancement simplifié** : Plus besoin de se rappeler de tous les fichiers .bat
- **🔧 Contrôles intégrés** : Gestion mémoire GPU, health checks, tests API
- **💻 CLI intégré** : Commandes health check et listing GPU directement
- **📊 Monitoring centralisé** : Accès à Qt, Web, Mobile depuis un endroit

---

## 🎛️ Interfaces Disponibles (via System Tray)

### 🖥️ **Qt Dashboard** (Recommandé pour monitoring)
- Interface native Windows
- Monitoring temps réel : CPU, RAM, SSD, GPU
- RTX 4060 Laptop GPU avec VRAM utilisée/totale
- Graphiques et barres de progression

### 🌐 **Dashboard Web Avancé** (Port 5000)
- Supervision cluster détaillée
- Informations nodes étendues (OS, CPU, RAM, VRAM, IP, Uptime)
- Interface web moderne avec auto-refresh
- Compatible tous navigateurs

### 📱 **Dashboard Mobile** (Port 5003)
- Interface responsive pour mobile/tablette
- Monitoring GPU avec barres de progression VRAM
- Tests API par sections (Health, System, GPU)
- Télémétrie détaillée multi-sections

### 🧪 **Debug Web Ultra** (Port 8080)
- Interface de débogage avancée
- Tests et diagnostics complets

---

## 🔧 Gestion Mémoire GPU (via System Tray)

Le menu **"Mémoire"** permet de :
- **Promouvoir 1er bloc** : Optimise l'allocation VRAM
- **Démonter 1er bloc** : Libère la mémoire GPU

---

## 💻 CLI Intégré (via System Tray)

Le menu **"CLI"** donne accès à :
- **Healthcheck** : Vérifie backend + énumération devices
- **Lister GPUs** : Affiche tous les GPUs VRAMancer détectés

---

## 🔍 Vérifications (via System Tray)

- **🔍 Vérifier API VRAMancer** : Test du port 5030
- Status avec notifications système

---

## 📋 Alternatives Directes (si nécessaire)

Si vous préférez lancer directement :

```bash
# Hub central avec menu
vramancer_hub.bat

# API seule (obligatoire pour les autres)
api_permanente.bat

# Interfaces individuelles
dashboard_qt.bat           # Qt natif
dashboard_web_avance.bat   # Web supervision
dashboard_mobile.bat       # Mobile responsive

# Tests spéciaux
test_cuda_ok.bat          # Test RTX 4060 + CUDA
diagnostic_cuda.py        # Diagnostic complet
```

---

## ✅ Workflow Recommandé

1. **Lancez** : `systray_vramancer.bat`
2. **Attendez** l'icône VRAMancer dans la barre des tâches
3. **Clic droit** sur l'icône
4. **Sélectionnez** l'interface souhaitée
5. **L'API se lance automatiquement** si nécessaire

---

## 🎮 RTX 4060 Laptop GPU - Fonctionnalités

Avec CUDA activé, toutes les interfaces détectent :
- ✅ **PyTorch CUDA 2.5.1**
- ✅ **RTX 4060 Laptop GPU**
- ✅ **Backend CUDA actif**
- ✅ **Monitoring VRAM temps réel**
- ✅ **Gestion mémoire optimisée**

---

## 🚀 Conclusion

**OUI, lancez toujours le System Tray !**

C'est votre **hub central** pour accéder à tout VRAMancer de manière simple et organisée.

---

*Compatible : Windows 10/11 + RTX 4060 Laptop GPU + PyTorch CUDA*