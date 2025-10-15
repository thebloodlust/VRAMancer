# 🍎 VRAMancer v1.1.0 - Installation macOS

## 📥 Téléchargement

### Option 1 : Depuis GitHub Releases (Recommandé)

1. Aller sur : https://github.com/thebloodlust/VRAMancer/releases/tag/v1.1.0
2. Télécharger : `VRAMancer-1.1.0-macOS.tar.gz`

### Option 2 : Avec curl

```bash
cd ~/Downloads
curl -L -O https://github.com/thebloodlust/VRAMancer/raw/main/VRAMancer-1.1.0-macOS.tar.gz
```

## 🚀 Installation

### Méthode 1 : Avec script automatique

```bash
# Aller dans le dossier de téléchargement
cd ~/Downloads

# Extraire l'archive
tar -xzf VRAMancer-1.1.0-macOS.tar.gz

# Rendre le script exécutable
chmod +x create_dmg_on_macos.sh

# Créer le DMG
./create_dmg_on_macos.sh

# Ouvrir le DMG
open VRAMancer-1.1.0-macOS.dmg

# Glisser VRAMancer.app dans Applications
```

### Méthode 2 : Installation manuelle

```bash
# Extraire
tar -xzf VRAMancer-1.1.0-macOS.tar.gz

# Créer le DMG manuellement
hdiutil create -volname "VRAMancer" \
    -srcfolder build_macos/dmg \
    -ov -format UDZO \
    VRAMancer-1.1.0-macOS.dmg

# Ouvrir
open VRAMancer-1.1.0-macOS.dmg
```

### Méthode 3 : Installation directe (sans DMG)

```bash
# Extraire
tar -xzf VRAMancer-1.1.0-macOS.tar.gz

# Copier directement dans Applications
sudo cp -R build_macos/VRAMancer.app /Applications/

# OU sans sudo (votre dossier utilisateur)
cp -R build_macos/VRAMancer.app ~/Applications/
```

## ❌ Résolution de Problèmes

### Erreur : "Archive non trouvée"

**Symptôme** :
```
❌ Archive VRAMancer-1.0.0-macOS.tar.gz non trouvée
```

**Solution** :
```bash
# 1. Vérifier le répertoire actuel
pwd

# 2. Lister les fichiers
ls -la | grep VRAMancer

# 3. Vous assurer d'être au bon endroit
cd ~/Downloads

# 4. Télécharger la bonne version
curl -L -O https://github.com/thebloodlust/VRAMancer/raw/main/VRAMancer-1.1.0-macOS.tar.gz
```

### Erreur : "getcwd: cannot access parent directories"

**Symptôme** :
```
shell-init: error retrieving current directory: getcwd: cannot access parent directories: Operation not permitted
```

**Causes possibles** :
1. Le répertoire a été supprimé/déplacé pendant que vous y étiez
2. Permissions insuffisantes

**Solution** :
```bash
# Retourner à un répertoire valide
cd ~/Downloads

# Relancer le script
./create_dmg_on_macos.sh
```

### Erreur : "Permission denied"

**Solution** :
```bash
# Rendre le script exécutable
chmod +x create_dmg_on_macos.sh

# Relancer
./create_dmg_on_macos.sh
```

### Erreur : "hdiutil not found"

**Solution** :
Vous n'êtes pas sur macOS. `hdiutil` est un outil macOS uniquement.

```bash
# Sur Linux/Windows, utilisez l'archive directement
tar -xzf VRAMancer-1.1.0-macOS.tar.gz
```

## 🎯 Quick Start Complet

```bash
# === COPIER-COLLER COMPLET ===

# 1. Téléchargement
cd ~/Downloads
curl -L -O https://github.com/thebloodlust/VRAMancer/raw/main/VRAMancer-1.1.0-macOS.tar.gz

# 2. Extraction
tar -xzf VRAMancer-1.1.0-macOS.tar.gz

# 3. Création DMG
chmod +x create_dmg_on_macos.sh
./create_dmg_on_macos.sh

# 4. Installation
open VRAMancer-1.1.0-macOS.dmg
# => Glisser VRAMancer.app dans Applications

# 5. Premier lancement
open /Applications/VRAMancer.app
```

## 🔐 Sécurité macOS

### "App cannot be opened because developer cannot be verified"

**Solution 1 : Autoriser l'app**
```bash
xattr -d com.apple.quarantine /Applications/VRAMancer.app
```

**Solution 2 : Via Préférences Système**
1. `Préférences Système` → `Sécurité et confidentialité`
2. Onglet `Général`
3. Cliquer `Ouvrir quand même`

### Donner les permissions

VRAMancer peut demander accès à :
- ✅ **Réseau** : Pour l'API (port 5030)
- ✅ **Fichiers** : Pour logs et configuration

## 📦 Contenu de l'Installation

```
/Applications/VRAMancer.app/
├── Contents/
│   ├── MacOS/
│   │   ├── VRAMancer          # Launcher
│   │   ├── api.py             # API production (nouveau v1.1.0)
│   │   ├── api_simple.py      # API simple (fallback)
│   │   ├── core/              # Modules principaux
│   │   │   └── production_api.py  # API robuste (nouveau)
│   │   ├── dashboard/         # Interfaces
│   │   ├── scripts/           # Scripts utilitaires
│   │   └── ...
│   ├── Resources/
│   │   ├── vramancer.png      # Icône
│   │   └── docs/              # Documentation
│   └── Info.plist             # Configuration macOS
```

## 🚀 Lancement

### Méthode 1 : Double-clic

```
/Applications/VRAMancer.app
```

### Méthode 2 : Ligne de commande

```bash
open /Applications/VRAMancer.app
```

### Méthode 3 : Depuis le terminal (mode production)

```bash
# Aller dans le bundle
cd /Applications/VRAMancer.app/Contents/MacOS

# Configurer (IMPORTANT pour production)
export VRM_AUTH_SECRET=$(openssl rand -hex 32)
export VRM_PRODUCTION=1

# Lancer
python3 api.py
```

## 🎮 Après Installation

### Vérifier que tout fonctionne

```bash
# Ouvrir le navigateur
open http://localhost:5030/health

# Résultat attendu :
# {"status": "healthy", "service": "vramancer-api", "version": "1.0.0"}
```

### Accéder aux interfaces

Une fois VRAMancer lancé :

1. **System Tray** : Icône dans la barre de menu (recommandé)
2. **Dashboard Web** : http://localhost:5030
3. **Dashboard Qt** : Interface graphique native
4. **Dashboard Mobile** : http://localhost:5003

## 📚 Documentation

Après installation, consultez :

- **`START_HERE.md`** : Guide de démarrage
- **`SECURITY_PRODUCTION.md`** : Sécurité (IMPORTANT !)
- **`MIGRATION_GUIDE.md`** : Migration dev → production
- **`README.md`** : Documentation complète

Ou en ligne : https://github.com/thebloodlust/VRAMancer

## 🆘 Support

Si vous rencontrez des problèmes :

1. **Logs** : `/Applications/VRAMancer.app/Contents/MacOS/logs/`
2. **Issues GitHub** : https://github.com/thebloodlust/VRAMancer/issues
3. **Documentation** : Voir les guides dans `/Applications/VRAMancer.app/Contents/Resources/docs/`

## 🎊 C'est tout !

VRAMancer v1.1.0 est maintenant installé sur votre Mac ! 🚀

**Prochaine étape** : Lire `START_HERE.md` pour configurer en mode production.
