# Guide Ultra-Débutant d'Installation VRAMancer (2025)

Ce guide tient la main d'un(e) néophyte complet(e). Aucune connaissance Linux / Python n'est supposée.

---
## 1. Télécharger le projet
- Ouvrez votre navigateur et allez sur : https://github.com/thebloodlust/VRAMancer
- Cliquez sur le bouton vert "Code" puis "Download ZIP".
- Une archive `VRAMancer-main.zip` est téléchargée.

## 2. Décompresser
- Windows : clic droit > "Extraire tout".
- macOS : double‑cliquez, un dossier est créé.
- Linux :
```bash
unzip VRAMancer-main.zip
```

Renommez si besoin le dossier en `VRAMancer` (évitez les espaces ou `(1)`).

## 3. Ouvrir un terminal dans ce dossier
- Windows : dans l'explorateur, tapez `cmd` dans la barre d'adresse et Entrée.
- macOS : clic droit > "Nouveau Terminal dans le dossier" (ou ouvrez Terminal puis `cd` vers le chemin).
- Linux : clic droit > Ouvrir dans un terminal.

## 4. Installation automatique (méthode simple)
Exécutez le script adapté :
- Windows :
```
installers\install_windows.bat
```
- Linux :
```bash
bash installers/install_linux.sh
```
- macOS :
```bash
bash installers/install_macos.sh
```

Ces scripts :
1. Installent les dépendances Python.
2. Configurent l'environnement local.
3. Lancent l'interface graphique d'installation.
4. Démarrent la découverte de cluster.
5. Effectuent un mini benchmark.

## 5. Lancer le dashboard
Après installation :
```bash
python -m vramancer.main --mode web
# ou
python -m vramancer.main --mode qt
```

## 6. Ajouter une autre machine (cluster plug-and-play)
1. Répétez l'installation sur une 2ᵉ machine.
2. Assurez‑vous qu'elles sont sur le même réseau (WiFi, Ethernet ou USB4).
3. Les nœuds apparaissent automatiquement dans le dashboard.

## 7. Vérifier la santé
```bash
curl -s http://localhost:5010/api/health
```
Réponse attendue : `{ "ok": true }`.

## 8. Mettre à jour
```bash
git pull
pip install -r requirements.txt --upgrade
```

## 9. (Optionnel) Activer les métriques
```bash
export VRM_METRICS_PORT=9108
```
Puis visiter `http://localhost:9108/metrics`.

## 10. Problèmes fréquents
| Symptôme | Cause probable | Solution rapide |
|----------|----------------|-----------------|
| Erreur tokenizers sous Windows | Rust absent | Installer https://rustup.rs puis relancer script |
| Torch GPU non détecté | Pilotes/CUDA manquants | Installer driver GPU + version CUDA/ROCm adaptée |
| Port déjà utilisé | Application déjà lancée | Tuer process ou changer port env (VRM_API_PORT) |
| ImportError psutil | psutil non installé | `pip install psutil` |

## 11. Désinstallation rapide
Supprimez simplement le dossier et (si créé) l'environnement virtuel `.venv`.

## 12. Support
Ouvrez une issue sur GitHub ou consultez `MANUEL_FR.md`.

---
Bonne orchestration ! 🚀
