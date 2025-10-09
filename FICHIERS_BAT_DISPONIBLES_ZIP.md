# 🎯 **Fichiers BAT Présents dans le ZIP GitHub**

## ✅ **Fichiers Disponibles dans Votre ZIP**

### 🚀 **simple_start.bat** (RECOMMANDÉ - Présent dans ZIP)
**Menu simple avec 3 choix :**
- [1] API + System Tray
- [2] API + Dashboard Qt  
- [3] API + Dashboard Web

**Utilisation :**
1. Double-clic sur `simple_start.bat`
2. Choisir option 1 pour System Tray
3. L'API se lance automatiquement puis le System Tray

### 🔧 **api_permanente.bat** (Présent dans ZIP)
**API permanente corrigée :**
- ✅ Utilise `api_simple.py` avec améliorations
- ✅ Relance automatique si crash
- ✅ Garde l'API en vie

### 🎛️ **systray_vramancer.bat** (Présent dans ZIP)
**System Tray basique :**
- ✅ Lance le System Tray avec icône
- ✅ Menu contextuel complet
- ⚠️ Nécessite API déjà lancée

## 🎯 **Ordre de Lancement pour Windows**

### **Option 1 : Tout Automatique (Recommandé)**
```batch
simple_start.bat
```
**Puis choisir [1]** → API + System Tray automatique

### **Option 2 : Manuel (2 étapes)**
```batch
1. api_permanente.bat     (laisser ouvert)
2. systray_vramancer.bat  (lance l'icône)
```

### **Option 3 : Autres Combinaisons**
- `simple_start.bat` → [2] pour API + Qt Dashboard
- `simple_start.bat` → [3] pour API + Web Dashboard

## 📋 **Autres Fichiers BAT Disponibles**

### **Dashboards Directs**
- `dashboard_qt.bat` - Qt dashboard seul
- `dashboard_web_avance.bat` - Web dashboard seul  
- `dashboard_mobile.bat` - Mobile dashboard seul

### **Lanceurs Spécialisés**
- `lanceur_systray.bat` - System Tray seul
- `lanceur_auto.bat` - Lancement automatique
- `vramancer_hub.bat` - Hub principal

## ✅ **Test de Fonctionnement**

Après lancement avec `simple_start.bat` → [1] :
- ✅ **API** active sur http://localhost:5030
- ✅ **Icône System Tray** dans barre des tâches
- ✅ **Clic droit** sur icône → Menu complet
- ✅ **GPU RTX 4060** avec affichage adaptatif MB/GB
- ✅ **Tous endpoints** fonctionnels

## 🎯 **Recommandation Finale**

**Utilisez `simple_start.bat`** car :
- ✅ Présent dans votre ZIP GitHub
- ✅ Menu simple et clair
- ✅ Lance API automatiquement
- ✅ Choix d'interface intégré
- ✅ Pas besoin de fichiers supplémentaires

**Double-clic et choisir [1] pour API + System Tray !** 🚀