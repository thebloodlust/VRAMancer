# VRAMancer User Manual

## Installation

- Windows: `installers/install_windows.bat`
- Linux  : `installers/install_linux.sh`
- macOS  : `installers/install_macos.sh`

## Getting Started & Setup

- Follow the graphical interface for installation and setup
- Plug in the machine (USB4, Ethernet, WiFi): node is auto-detected
- Master is chosen by performance (can be overridden)

## Orchestration & Usage

- Qt/Tk/Web/CLI dashboard for monitoring and control
- Universal plug-and-play, auto-sensing, dynamic clustering
- VRAM/CPU aggregation, intelligent block routing
- Manual master/slave override available

## Main Commands

- Launch dashboard: `python -m vramancer.main --mode qt` (or tk/web/cli)
- Launch cluster master: `python core/network/cluster_master.py`
- Discover nodes: `python core/network/cluster_discovery.py`
- Aggregation & routing: `python core/network/resource_aggregator.py`

## Examples

- See `EXAMPLES_CLUSTER.md` for concrete use cases

## Support

- Full documentation: [RELEASE.md](RELEASE.md), [EXAMPLES.md](EXAMPLES.md), [CONTRIBUTING.md](CONTRIBUTING.md)
- For any questions, open an issue on GitHub
