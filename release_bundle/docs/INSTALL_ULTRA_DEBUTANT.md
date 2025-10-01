# Guide d’installation ultra-débutant VRAMancer

## 1. Prérequis
- Avoir Python 3 installé (Linux/macOS/Windows)
- Avoir extrait le dossier `release_bundle`

## 2. Installation

### Linux
- Double-cliquez ou lancez dans un terminal :
  ```bash
  bash installers/install_linux.sh
  ```
- Ou installez le paquet `.deb` :
  ```bash
  sudo dpkg -i vramancer.deb
  ```

### macOS
- Ouvrez un terminal et lancez :
  ```bash
  bash installers/install_macos.sh
  ```

### Windows
- Double-cliquez sur `installers/install_windows.bat` ou lancez dans l’invite de commande.

## 3. Lancement
- Activez l’environnement virtuel si demandé (`source .venv/bin/activate` ou `.\.venv\Scripts\activate`)
- Lancez le dashboard ou la CLI :
  ```bash
  python dashboard/dashboard_qt.py
  python cli/dashboard_cli.py
  ```

## 4. Support
- Consultez le README ou docs/ pour plus d’exemples et d’options.
