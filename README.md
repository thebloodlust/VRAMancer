<div align="center">
  <h1>🚀 VRAMancer</h1>
  <p><b>The Heterogeneous AI Swarm / L'Essaim IA Asynchrone Zero-Trust</b></p>
  <p><i>Orchestrate massive LLMs efficiently across mismatched hardware.</i></p>
  <p>
    <img src="https://img.shields.io/badge/Hardware-Asymmetric%20GPU%20Support-blue" alt="Hardware">
    <img src="https://img.shields.io/badge/Network-Zero--Copy%20Rust%20P2P-purple" alt="Network">
    <img src="https://img.shields.io/badge/Performance-Zero--Copy%20%7C%20Async-success" alt="Performance">
  </p>
</div>

---

> **🚀 Hack The Hardware Limitations:** Break Nvidia's artificial limits. Combine your main GPU (e.g. RTX 3090 24GB) and your old card (e.g. RTX 5070 Ti 16GB) to run pure Tensor Parallelism natively without crashing your VRAM. Maintain desktop CPU speeds (17+ tok/s on Qwen 14B) via our native `Rust tokio` Resizable-BAR DMA wrapper bypassing `NVLink` locks!

### 🔥 Fonctionnalités Cœurs (Architecture Hautes-Performances) :

- **Asymmetric GPU Tensor Parallelism :** Divisez de gigantesques modèles (14B, 32B, 70B paramètres) sur des grappes de GPUs de tailles différentes sans Out-Of-Memory (OOM) via VLLM adaptatif (`gpu_memory_utilization`).
- **Rust Zero-Copy Network Bridge :** Remplacement ultra-rapide des librairies P2P/NVLink censurées par un moteur Rust Asynchrone TCP-loopback (`tokio`) qui s'interface au PCIe, relâchant complètement le GIL Python.
- **VRAMancer Cyberpunk Web Dashboard :** Visualisez la distribution mémoire de votre cluster et discutez directement avec l'IA via un Chat Web intégré aux capacités "OpenAI-compatible" de l'API originelle `/v1/models/`.
- **Mémoire Hiérarchique à 6 Niveaux :** Transfert fluide et intelligent de la VRAM (L1), vers NVLink/P2P (L2), vers la Pinned Memory classique/DRAM (L3), jusqu'au Swap NVMe ultra-rapide (L4) si nécessaire.

---

## 🟢 Installation Simplifiée (Easy Install)

VRAMancer supporte nativement un très grand nombre de matériels (NVIDIA, AMD ROCm, Apple MPS). Nous avons packagé le projet pour un démarrage immédiat en 3 lignes.

### Linux / Ubuntu (Recommandé)

```bash
# 1. Clonez l'essaim
git clone https://github.com/thebloodlust/VRAMancer.git
cd VRAMancer

# 2. Lancez le script d'installation automatique
# Il va créer le .venv, installer VLLM, et compiler le bridge P2P Rust (maturin) dynamiquement !
./Install.sh

# 3. Lancez le serveur API / Dashboard local
source .venv/bin/activate
VRM_CORS_ORIGINS="*" python3 -m vramancer.main serve --host "0.0.0.0" --port 5000
```
*(Vous pouvez désormais accéder à votre Dashboard Asymétrique en ouvrant `http://localhost:8500/` dans votre navigateur Web !)*

---

## 🚑 Mécanisme de Survie : Nettoyage d'Urgence (Recovery)

L'Inférence distribuée sur GPU asymétriques manipule les buffers de VRAM au plus bas niveau. Si vous forcez un arrêt serveur ou chargez un modèle trop lourd entraînant un redoutable `CUDA error: out of memory`, la VRAM peut se retrouver figée par des processus fantômes.

**Pas besoin de redémarrer le PC. Utilisez notre script d'urgence de la mort.**

```bash
# Dans le dossier de VRAMancer :
chmod +x rescue_vram.sh
./rescue_vram.sh
```

**Ce script de niveau Kernel va :**
1. Tuer brutalement n'importe quel démon `multiproc_executor`, `vllm` ou de worker en Python zombie.
2. Vider les Caches Mémoires système RAM & Swap (Sysvm Root).
3. Interroger le bus NVIDIA (`nvidia-smi`) pour forcer l'extraction de tous les tensors bloqués via `fuser -k -9 /dev/nvidia*`.
4. Relâcher 100% de la matrice VRAM pour un reset à neuf.

---

*(Pour les détails profonds du moteur interne asynchrone VRAMancer ou les requêtes API d'Architecture distribuée, lisez notre [📝 docs/architecture.md](./docs/architecture.md))*
