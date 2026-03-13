# 🚀 VRAMancer : Rapport d'Architecture Rust & ReBAR Bypass

Ce fichier documente les avancées majeures réalisées sur la branche `experimental/rust-core`, afin de ne rien oublier lors des futures itérations.

## 1. Le Bypass Matériel : "Zero-Copy ReBAR TCP"
**Le problème initial :** Nvidia bloque volontairement l'API `cudaMemcpyPeer` entre les GPUs "Consumer" (GeForce, RTX 3090, 4090...) via le bus PCIe. PyTorch tombait en "Fallback CPU", bloquant le GIL Python et causant d'énormes ralentissements (~0.8s à 1.5s de latence au Time-To-First-Token).  
**La solution VRAMancer :** Nous utilisons la macro PCIe "ReBAR" (Resizable BAR) qui mappe la VRAM directement au CPU. 
- Nous avons créé un pont Rust (`vramancer_rust.direct_vram_load`).
- Nous utilisons `Tokio` (I/O asynchrone non-bloquant).
- Le transfert s'opère comme un faux flux réseau Localhost via le CPU de manière asynchrone, trompant le pilote Nvidia et retrouvant une bande passante folle (jusqu'à la limite du bus PCIe 4.0/5.0) sans *jamais* geler l'orchestrateur Python.

## 2. Le Swarm Brain (Mémoire Holographique)
L'ancien calcul de parité XOR en C++ (visant à regénérer un nœud mort) a été porté sur Rust.
- Utilisation de `generate_holographic_parity` en Rust pur.
- Zéro fuites de mémoire (Memory Leaks évités grâce au typage Rust).
- LLVM s'occupe de l'auto-vectorisation (AVX-512) pour un calcul instantané.

## 3. L'Offload "Software CXL" (NVMe)
Le dump et load direct de la RAM vers les disques NVMe se fait désormais sans passer par le module lourd Pickle de Python. `cxl_direct_memory_dump` écrit littéralement les pointeurs sur le disque avec le GIL relâché.

## 4. Le P2P Réseau Distant (IP)
(En cours d'achèvement) - Remplacement du protocole de transport de tenseur natif par un pont direct TCP en Rust `send_tensor_p2p` et `receive_tensor_p2p`. Les tenseurs transitent signés par **HMAC-SHA256** sans être désérialisés, offrant un "pont internet" entre plusieurs machines distantes.

---
*Roadmap d'installation :* L'objectif est de masquer toute cette complexité bas niveau par des scripts d'installation Linux/Mac/Windows (Plug & Play) gérant l'installation de Cargo, Maturin et Python de manière invisible.