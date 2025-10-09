# 🚀 VRAMancer - Lancement Rapide avec Améliorations

## ✅ **Toutes les Améliorations Incluses**

- 🎮 **GPU Adaptatif MB/GB** : RTX 4060 affiché précisément
- 🌐 **Endpoints complets** : `/api/nodes`, `/api/gpu`, `/api/system`
- 📱 **Mobile sans 404** : Tous les endpoints fonctionnels
- 🔧 **Dashboard Qt** : VRAM adaptatif selon usage

## 🎯 **Lancement Simple**

### **Windows**
```bash
launch_vramancer.bat
```

### **Linux/Mac**  
```bash
./launch_vramancer.sh
```

### **Manuel (toutes plateformes)**
```bash
# 1. API (obligatoire)
python api_simple.py

# 2. Interface au choix
python systray_vramancer.py                    # System Tray Hub
python dashboard/dashboard_qt.py               # Qt avec MB/GB adaptatif
python dashboard/dashboard_web_advanced.py     # Web avec détails nodes
python mobile/dashboard_mobile.py              # Mobile sans 404
```

## 🎮 **Test RTX 4060 Laptop GPU**

Avec les améliorations, vous verrez :
- **Usage < 1GB** → `234 MB / 8.2 GB` (précis)  
- **Usage ≥ 1GB** → `2.3 GB / 8.2 GB` (lisible)

## 🌐 **URLs Dashboards**

- **Web Advanced** : http://localhost:5000
- **Mobile** : http://localhost:5003  
- **API** : http://localhost:5030

## 🔧 **Dépendances**

```bash
pip install flask requests psutil torch
pip install PyQt5  # Pour dashboard Qt
```

## ⚡ **Lancement Ultra-Rapide**

1. **Double-clic** sur `launch_vramancer.bat` (Windows)
2. **System Tray** apparaît dans la barre des tâches
3. **Clic droit** → Choisir votre dashboard

## ✅ **Vérification Améliorations**

### **GPU Adaptatif Fonctionnel ?**
- Dashboard Qt → VRAM en MB pour petits usages
- Dashboard Mobile → Même logique adaptatif

### **Web Advanced Complet ?**
- Détails nodes → OS, CPU, VRAM, GPU, Backend, IP
- Grid layout → Informations organisées

### **Mobile Sans Erreurs ?**
- Plus de croix rouge 404
- GPU et système affichés correctement

🎉 **Tout est prêt pour votre RTX 4060 Laptop GPU !**