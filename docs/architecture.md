# VRAMancer - Architecture Technique Profonde

Ce document décrit les choix radicaux, l'architecture interne et le pipeline de données qui permettent à VRAMancer d'être un orchestrateur GPU multi-nœuds apte à l'inférence asynchrone intensive.

## 1. Mémoire Hiérarchique à 6 Niveaux (Tiered Memory)
Au cœur de l'orchestrateur se trouve le module `HierarchicalMemory`. Il ne considère pas seulement la VRAM locale, mais orchestre une cascade dynamique :
1. **L1 : VRAM Hautes Performances** (GPU local 0)
2. **L2 : VRAM P2P** (GPU local 1 via NVLink/PCIe)
3. **L3 : DRAM Host (CPU)** (Pinned memory / Zero-Copy)
4. **L4 : Swap NVMe ultra-rapide**
5. **L5 : VRAM Distante** (Grappe de serveurs Zero-Trust / Réseau local)
6. **L6 : Stockage Distant / Hub**

Les tenseurs et les blocs KVCache glissent de manière transparente entre ces niveaux grâce à nos heuristiques basées sur des algorithmes de "Heat" et "TTL" (LRU adapté).

## 2. Inférence Asynchrone Native (GIL Bypassed)
VRAMancer ne se contente pas de requêtes synchrones :
- **vLLM Integration** : Abandon de l'ancien objet `LLM` bloquant au profit du `AsyncLLMEngine`. Cela active le mécanisme de **Continuous Batching**, permettant à Python de ne jamais bloquer la boucle d'événements locale tout en laissant le moteur C++ vLLM empiler les requêtes en continu.
- **TensorRT Zero-Copy** : Utilisation intensive des pointeurs mémoires physiques (`inputs.data_ptr()`) passés dans des `streams` CUDA asynchrones (`execute_async_v2`). Le CPU n'orchestre que l'adresse, la copie mémoire VRAM-RAM-VRAM est pulvérisée.
- **Ollama Async** : Les accès TCP utilisent nativement `aiohttp` pour ne pas épuiser le pool de threads de Flask / de l'orchestrateur.

## 3. Topologie Réseau P2P & Sécurité Zero-Trust
Pour un cluster réparti, VRAMancer n'est pas un système fermé naïf. 
- La sérialisation inter-noeuds transite par le module `BlockRouter`.
- Chaque bloc Tensor partagé par TCP est enveloppé d'une signature cryptographique **HMAC-SHA256**.
- Tout payload reçu est impitoyablement vérifié avant d'exécuter `pickle.loads()`, empêchant de fait l'exécution de code à distance (RCE) typique des implémentations PyTorch. 
- Les WebSockets de l'infrastructure GPU WebGPU (`core/network/webgpu_node.py`) intègrent une vérification asynchrone des tokens JWT avant d'allouer des workers.

## 4. Découverte de Modèles "Docker-like" 
L'expérience développeur (CLI) s'efforce d'être à l'IA ce que Docker est aux conteneurs :
- Moteur de découverte (via la librairie terminale `Rich`) sur Hub API (`vramancer hub model/name`).
- L'API inspecte pour vous les formats avancés : **NVFP4 (Blackwell), GPTQ, AWQ, GGUF, INT8**.

## 5. La Degradation Gracieuse
Les environnements asynchrones sont sujets aux *Deadlocks* (par exemple l'attente infinie sur un Node réseau déchu, ou un thread CUDA attendant une synchronisation de stream). 
VRAMancer dégrade silencieusement les blocs défectueux (Timeout `socket`, `finally: stream.synchronize()`) pour empêcher une faute de segmentation destructrice (segfault) de se propager du C++ jusqu'à Python.