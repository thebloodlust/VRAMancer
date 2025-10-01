# Manuel d’utilisation VRAMancer

## Installation

- Windows : `installers/install_windows.bat`
- Linux   : `installers/install_linux.sh`
- macOS   : `installers/install_macos.sh`

## Démarrage et configuration

- Suivez l’interface graphique pour l’installation et la configuration
- Branchez la machine (USB4, Ethernet, WiFi) : le nœud est détecté automatiquement
- Le master est choisi selon la performance (modifiable)

## Orchestration et usage

- Dashboard Qt/Tk/Web/CLI pour monitoring et contrôle
- Plug-and-play universel, auto-sensing, clustering dynamique
- Agrégation VRAM/CPU, routage intelligent des blocs
- Override manuel du master/slave possible

## Commandes principales

- Lancer le dashboard : `python -m vramancer.main --mode qt` (ou tk/web/cli)
- Lancer le cluster master : `python core/network/cluster_master.py`
- Lancer la découverte de nœuds : `python core/network/cluster_discovery.py`
- Agrégation et routage : `python core/network/resource_aggregator.py`

## Exemples

- Voir `EXAMPLES_CLUSTER.md` pour des cas d’usage concrets

## Support

- Documentation complète : [RELEASE.md](RELEASE.md), [EXAMPLES.md](EXAMPLES.md), [CONTRIBUTING.md](CONTRIBUTING.md)
- Pour toute question, ouvrez une issue sur GitHub
