# 🍎 Guide d'installation VRAMancer sur macOS

## 📦 Ce que vous avez

Vous avez maintenant **2 fichiers** :

1. ✅ **`VRAMancer-1.0.0-macOS.tar.gz`** (1.6 MB) - Archive contenant l'application
2. ✅ **`create_dmg_on_macos.sh`** - Script pour créer le .dmg sur macOS

## 🚀 Installation sur votre Mac

### Étape 1 : Transférer les fichiers sur votre Mac

Copiez ces deux fichiers sur votre Mac (USB, réseau, email, etc.)

### Étape 2 : Extraire et créer le .dmg

Ouvrez un Terminal sur votre Mac et exécutez :

````bash
cd ~/Downloads  # Ou le dossier où vous avez copié les fichiers

# Rendre le script exécutable
chmod +x create_dmg_on_macos.sh

# Créer le .dmg
./create_dmg_on_macos.sh
````

Cela va créer : **`VRAMancer-1.0.0-macOS.dmg`**

### Étape 3 : Installer VRAMancer

1. **Double-cliquez** sur `VRAMancer-1.0.0-macOS.dmg`
2. Une fenêtre s'ouvre avec :
   - L'icône **VRAMancer.app**
   - Un dossier **Applications**
   - Un fichier **README.txt**
3. **Glissez-déposez** VRAMancer.app dans le dossier Applications
4. Fermez la fenêtre du DMG
5. Éjectez le volume VRAMancer (Cmd+E ou clic droit → Éjecter)

### Étape 4 : Premier lancement

1. Ouvrez **Launchpad** ou allez dans `/Applications`
2. **Double-cliquez** sur VRAMancer
3. Si macOS affiche "Application non vérifiée" :
   - Allez dans **Préférences Système** → **Confidentialité et sécurité**
   - Cliquez sur **Ouvrir quand même**
   - Ou : clic droit sur VRAMancer.app → **Ouvrir**

**Premier démarrage** : L'installation des dépendances Python prend 2-5 minutes.

## 🎮 Utilisation

### Interfaces disponibles

Une fois lancé, VRAMancer offre plusieurs interfaces :

#### 1. **System Tray** (recommandé)
- Icône dans la barre de menu (en haut à droite)
- Clic → Menu avec toutes les options
- Nécessite PyQt5 (installé automatiquement)

#### 2. **Dashboard Web**
- Ouvre automatiquement dans votre navigateur
- URL : http://localhost:5030
- Interface web moderne et responsive

#### 3. **Dashboard Qt**
- Interface graphique native macOS
- Plus rapide et intégrée au système

#### 4. **Dashboard Mobile**
- Design responsive
- URL : http://localhost:5003

### Commandes pratiques

````bash
# Lancer manuellement depuis le Terminal
cd /Applications/VRAMancer.app/Contents/MacOS
source .venv/bin/activate
python3 api_simple.py &
python3 systray_vramancer.py

# Ou avec le script intégré
./vrm_start.sh
````

## 🔧 Configuration avancée

### Installer PyQt5 pour le System Tray

Si le System Tray ne s'affiche pas :

````bash
# Aller dans l'application
cd /Applications/VRAMancer.app/Contents/MacOS

# Activer l'environnement virtuel
source .venv/bin/activate

# Installer PyQt5
pip install PyQt5
````

### Fichiers de configuration

- **Config** : `~/.vramancer/config.yaml`
- **Logs** : `~/.vramancer/logs/`
- **API** : Port 5030 (modifiable dans config.yaml)

## 📚 Documentation

Documentation complète dans l'application :

````bash
cd /Applications/VRAMancer.app/Contents/MacOS
open README.md          # Documentation principale
open MANUEL_FR.md       # Guide en français
open MANUAL_EN.md       # English manual
````

Ou sur le web : http://localhost:5030/docs (quand l'API tourne)

## 🛠️ Dépannage

### L'API ne démarre pas

````bash
# Vérifier si le port 5030 est occupé
lsof -i :5030

# Tuer le processus qui occupe le port
kill -9 <PID>
````

### Python non trouvé

Installez Python 3 depuis : https://www.python.org/downloads/macos/

### PyQt5 non trouvé

````bash
pip3 install PyQt5
````

### Erreur de permissions

````bash
# Réparer les permissions
cd /Applications
sudo chown -R $(whoami) VRAMancer.app
chmod +x VRAMancer.app/Contents/MacOS/VRAMancer
````

### Logs et diagnostics

````bash
# Voir les logs
cat ~/.vramancer/logs/api.log
cat ~/.vramancer/logs/app.log

# Voir les processus VRAMancer
ps aux | grep vramancer
````

## 🎯 Cas d'usage

### 1. Orchestration IA multi-GPU

- Répartition automatique de la charge VRAM
- Support CUDA, Metal (Apple Silicon), ROCm
- Optimisation dynamique

### 2. Clustering distribué

- Découverte automatique des nœuds
- Répartition intelligente des tâches
- Monitoring temps réel

### 3. Dashboard de supervision

- Métriques GPU en temps réel
- Heatmap VRAM
- Alertes et notifications

### 4. Mobile / Remote

- Contrôle depuis smartphone
- API REST complète
- WebSocket temps réel

## 📞 Support

- **Documentation** : http://localhost:5030/docs
- **GitHub** : https://github.com/thebloodlust/VRAMancer
- **Issues** : https://github.com/thebloodlust/VRAMancer/issues
- **Logs** : `~/.vramancer/logs/`

## 📋 Checklist d'installation

- [ ] Python 3.9+ installé
- [ ] VRAMancer.app dans /Applications
- [ ] Premier lancement effectué (dépendances installées)
- [ ] API accessible sur http://localhost:5030/health
- [ ] Interface choisie (Tray, Web, Qt, Mobile)
- [ ] PyQt5 installé (optionnel, pour System Tray)
- [ ] Documentation lue (README.md)

## 🎉 C'est parti !

Vous êtes prêt à utiliser VRAMancer pour orchestrer vos modèles IA !

**Commande rapide** :

````bash
# Lancer VRAMancer
open -a VRAMancer

# Ou depuis Launchpad
# Chercher "VRAMancer" et cliquer
````

---

**Version** : 1.0.0  
**Date** : Octobre 2025  
**Licence** : MIT
