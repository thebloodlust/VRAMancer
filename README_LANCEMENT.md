# 🚀 VRAMancer - Guide de Lancement Final

## 📦 Après Téléchargement du ZIP

### Windows
```batch
# Option 1: Complet (RECOMMANDÉ)
START_VRAMANCER.bat

# Option 2: Simple et rapide
vrm_start.bat

# Option 3: API uniquement
api_permanente.bat
```

### Mac/Linux
```bash
# Option 1: Complet (RECOMMANDÉ)
./START_VRAMANCER.sh

# Option 2: Simple et rapide  
./vrm_start.sh

# Option 3: Manuel
./launch_vramancer.sh
```

## ✅ Toutes les Corrections Incluses

### Mobile Dashboard
- ❌ Erreur "gpuHtml not defined" → ✅ CORRIGÉ
- ❌ Erreurs 404 récupération → ✅ CORRIGÉ
- ✅ GPU adaptatif MB/GB selon usage RTX 4060

### Web Dashboard Advanced
- ❌ Détails nodes manquants → ✅ CORRIGÉ
- ✅ Informations complètes: OS, CPU, GPU, VRAM, réseau
- ✅ Endpoint /api/nodes fonctionnel

### Qt Dashboard
- ❌ Affichage VRAM imprécis → ✅ CORRIGÉ
- ✅ Affichage adaptatif: MB pour <1GB, GB pour ≥1GB
- ✅ Monitoring RTX 4060 Laptop GPU précis

### Windows Launchers
- ❌ Caractères spéciaux non reconnus → ✅ CORRIGÉ
- ❌ Erreurs 'VRAMancer' commande → ✅ CORRIGÉ
- ✅ 3 launchers propres sans erreurs d'encodage

## 🎯 Recommandations

### Windows (après extraction ZIP)
1. **START_VRAMANCER.bat** - Interface complète avec choix 1-5
2. **vrm_start.bat** - Version minimaliste 1-4 
3. Choisir **[1] System Tray** pour l'expérience complète

### Mac (après extraction ZIP)
1. **./START_VRAMANCER.sh** - Interface complète avec détection auto Python
2. **./vrm_start.sh** - Version rapide
3. Choisir **[1] System Tray** ou **[6] Console Hub** sans GUI

## 🎮 Résultats Attendus

Après lancement avec RTX 4060 Laptop GPU:
- **API** active sur http://localhost:5030
- **System Tray** avec icône et menu complet  
- **Qt Dashboard** avec VRAM adaptive MB/GB
- **Web Dashboard** sur http://localhost:5000 avec détails nodes
- **Mobile Dashboard** sur http://localhost:5003 sans erreurs

## 🧹 Nettoyage Effectué

- **62 fichiers .bat** réduits à **3 optimaux**
- **Tous caractères spéciaux** remplacés par ASCII
- **Encodage Windows** compatible tous systèmes
- **Scripts Mac/Linux** avec détection Python auto
- **Documentation** simplifiée et claire

## ⚡ Lancement Ultra-Rapide

**Windows:** Double-clic `START_VRAMANCER.bat` → [1]  
**Mac:** Double-clic `START_VRAMANCER.sh` → [1]  
**Résultat:** VRAMancer complet avec RTX 4060 en 10 secondes ! 🚀