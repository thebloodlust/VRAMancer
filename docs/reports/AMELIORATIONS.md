# VRAMancer — Améliorations intéressantes à apporter

> Idées d'évolution classées par valeur ajoutée, distinctes des corrections (voir CORRECTIONS.md).
> Focus : ce qui ferait passer VRAMancer d'un projet ambitieux à un outil compétitif.

---

## Tier 1 — Améliorations à forte valeur

### 1. Passer l'API en async natif (FastAPI/Quart)

**Situation actuelle :** Flask synchrone + `ThreadPoolExecutor` pour simuler l'async. Chaque requête bloque un thread OS.

**Amélioration :** Migrer vers FastAPI (ou Quart pour rester Flask-compatible) :
- **Streaming natif** : `StreamingResponse` + `async yield` pour le token-by-token
- **Concurrence** : 1000+ requêtes simultanées sans pool de threads
- **WebSocket natif** : plus besoin de `flask-socketio` + eventlet
- **OpenAPI auto** : documentation Swagger générée automatiquement

**Impact :** Latence réduite de 30-50% sur les requêtes concurrentes. Compatibilité native avec vLLM (qui est déjà async).

---

### 2. Connection pool Tokio persistante (Rust)

**Situation actuelle :** Chaque `send_tensor_p2p()` crée un nouveau runtime Tokio. Le coût de création (~1ms) est significatif sur des transferts fréquents.

**Amélioration :**
```rust
// Runtime persistant partagé par tous les transferts
lazy_static! {
    static ref RUNTIME: Runtime = Runtime::new().unwrap();
}

// Pool de connexions TCP pré-ouvertes
struct ConnectionPool {
    connections: DashMap<SocketAddr, TcpStream>,
}
```

**Impact :** Latence P2P réduite de ~1ms à ~0.1ms pour les transferts répétés entre les mêmes nœuds.

---

### 3. Streaming model loading (layer-by-layer)

**Situation actuelle :** `AutoModel.from_pretrained()` charge le modèle entier en RAM avant de le splitter. Pour un modèle 70B, ça demande ~140 GB de RAM temporaire.

**Amélioration :** Charger et distribuer couche par couche :
```python
for layer_name, tensor in safetensors.torch.load_file(path, device="meta"):
    target_gpu = placement_plan[layer_name]
    tensor = tensor.to(f"cuda:{target_gpu}")
    model.load_state_dict({layer_name: tensor}, strict=False)
```

Avec `device_map="auto"` de HuggingFace Accelerate, c'est partiellement supporté. L'amélioration serait d'intégrer le `model_splitter` VRAM-proportionnel directement dans le flux de chargement.

**Impact :** Chargement de modèles 70B+ sur machines avec 32 GB RAM. Game changer pour les setups prosumer.

---

### 4. Mode "zero-config" avec auto-détection complète

**Situation actuelle :** Il faut configurer manuellement les GPU, les backends, les tokens, etc.

**Amélioration :** Un mode `vramancer auto` qui :
1. Détecte tous les GPU (CUDA/ROCm/MPS) et leur VRAM libre
2. Choisit le backend optimal (vLLM si dispo, sinon HF)
3. Calcule le split automatiquement
4. Génère un token aléatoire et l'affiche
5. Lance l'API sur le premier port libre

```bash
$ vramancer auto --model meta-llama/Llama-3-70B
🔍 Found: RTX 3090 (24GB) + RTX 5070 Ti (16GB) = 40GB total
📐 Split: 60%/40% (VRAM-proportional)
🔧 Backend: HuggingFace (vLLM not found)
🔑 Token: vrm_a7b3c9d2e1f4
🚀 API: http://localhost:8000/v1/completions
```

**Impact :** Accessibilité massive. Le principal concurrent (Ollama) est populaire exactement pour cette raison.

---

### 5. Support quantization au chargement (GPTQ/AWQ/GGUF)

**Situation actuelle :** Le compressor fait de la quantization INT8/INT4 post-chargement. Mais les modèles quantifiés (GPTQ, AWQ, GGUF) ne sont pas supportés nativement.

**Amélioration :** Intégrer `auto-gptq`, `autoawq`, ou `llama-cpp-python` comme backends de quantization :
```python
# Dans select_backend() :
if model_name.endswith("-GPTQ"):
    return GPTQBackend(model_name)
elif model_name.endswith("-GGUF"):
    return GGUFBackend(model_name)
```

**Impact :** Les modèles GGUF sont le standard de facto pour l'inférence locale. Sans ce support, VRAMancer rate 80% du marché prosumer.

---

### 6. KV cache offloading intelligent

**Situation actuelle :** `paged_attention.py` gère un KV cache paginé mais tout reste en VRAM. Quand la VRAM est pleine, l'inférence échoue.

**Amélioration :** Offloader les pages KV les moins récentes vers la RAM (pinned) puis le NVMe, avec prefetch prédictif :
```
VRAM (hot) ──evict──> RAM pinned (warm) ──evict──> NVMe (cold)
                       ◄──prefetch──            ◄──prefetch──
```

Le `hierarchical_memory.py` a déjà cette architecture ! L'amélioration serait de le connecter au `paged_attention.py` qui actuellement ne l'utilise pas.

**Impact :** Contextes 128K+ tokens sur des GPU 24 GB. Actuellement limité par la VRAM pour les longs contextes.

---

## Tier 2 — Améliorations utiles

### 7. Dashboard temps réel avec WebSocket push

**Situation actuelle :** Le dashboard web poll toutes les 2 secondes. Le dashboard CLI utilise curses.

**Amélioration :** WebSocket push (déjà partiellement implémenté via flask-socketio) pour :
- Timeline d'inférence en temps réel (quel GPU exécute quelle couche)
- Heatmap VRAM animée
- Graphe de topologie P2P dynamique
- Alertes push (pas de polling)

---

### 8. Plugin system pour les backends

**Situation actuelle :** Ajouter un backend nécessite de modifier `select_backend()` dans `backends.py`.

**Amélioration :** Système de plugins via entry_points :
```toml
# pyproject.toml d'un plugin tiers
[project.entry-points."vramancer.backends"]
my_backend = "my_package:MyBackend"
```

```python
# Dans VRAMancer :
for ep in importlib.metadata.entry_points(group="vramancer.backends"):
    BACKENDS[ep.name] = ep.load()
```

**Impact :** Écosystème ouvert. Les utilisateurs peuvent ajouter MLX, CoreML, OpenVINO, etc. sans forker le projet.

---

### 9. Batch inference avec continuous batching réel

**Situation actuelle :** `continuous_batcher.py` est tronqué. L'inférence est séquentielle (une requête à la fois).

**Amélioration :** Implémenter un vrai continuous batching (inspiré de vLLM/TGI) :
- Fusion de tokens de plusieurs requêtes dans un seul forward pass
- Scheduling preemptif (interrompre une requête longue pour servir une courte)
- Padding dynamique (pas de gaspillage sur les séquences courtes)

**Impact :** Throughput ×3-10 sur les workloads multi-utilisateurs.

---

### 10. Compléter l'intégration XDP/eBPF

**Situation actuelle :** `csrc/aitp_xdp_bypass.c` est un PoC qui intercepte les paquets mais ne fait pas le DMA vers le GPU.

**Amélioration :** Implémenter le chemin complet :
1. XDP intercepte le paquet AITP sur le NIC
2. BPF map stocke l'adresse VRAM cible
3. Le driver nvidia_peermem fait le DMA NIC→GPU
4. Un BPF ringbuffer notifie le userspace que le tenseur est arrivé

**Impact :** Latence réseau ~0.5µs au lieu de ~10µs. Utile uniquement pour les clusters InfiniBand/RoCE haute performance.

---

### 11. Auto-scaling horizontal

**Situation actuelle :** Le cluster est statique. Ajouter un nœud nécessite une intervention manuelle.

**Amélioration :** Avec `cluster_discovery.py` (mDNS) + `wake_on_inference.py`, implémenter :
- Auto-discovery de nouveaux nœuds GPU
- Migration automatique de layers vers les nouveaux nœuds
- Scale-down : mettre en veille les nœuds inactifs après N minutes

---

### 12. Export ONNX / CoreML / OpenVINO

**Situation actuelle :** Uniquement inférence via PyTorch.

**Amélioration :** Ajouter un pipeline d'export :
```bash
vramancer export --model llama-3 --format onnx --output ./model.onnx
vramancer export --model llama-3 --format coreml --output ./model.mlpackage
```

**Impact :** Permet de préparer les modèles pour le déploiement edge (mobile, IoT) — complémentaire au `edge_iot.py`.

---

## Tier 3 — Améliorations exploratoires

### 13. Intégration MLX pour Apple Silicon

Apple MLX est optimisé pour les puces M1-M4 (unified memory). Actuellement, VRAMancer utilise PyTorch MPS qui sous-exploite le hardware Apple.

Un `MLXBackend` qui utilise directement MLX pourrait donner ×2-3 de throughput sur Mac.

---

### 14. Multi-modèle concurrent

Charger plusieurs modèles sur les mêmes GPU (ex: un petit modèle pour le routing + un gros pour la génération). Le `scheduler.py` devrait pouvoir gérer des slots mémoire par modèle.

---

### 15. Fédération de clusters (multi-datacenter)

Le `Swarm Ledger` a les bases d'une économie P2P. L'étendre pour permettre la fédération de clusters distants (ex: ton cluster maison + celui d'un ami) avec :
- Accounting des crédits GPU
- Priorité locale > réseau
- Chiffrement E2E des tenseurs

---

### 16. Intégration LoRA/QLoRA hot-swap

Charger des adaptateurs LoRA à chaud, sans recharger le modèle de base :
```bash
vramancer lora load ./my-adapter --base llama-3
vramancer lora unload my-adapter
```

Très demandé pour les workloads multi-tenant (un LoRA par client).

---

## Matrice valeur/effort

| # | Amélioration | Valeur | Effort | Priorité |
|---|-------------|--------|--------|----------|
| 4 | Mode zero-config | ★★★★★ | ★★☆☆☆ | **Faire en premier** |
| 5 | Support GPTQ/AWQ/GGUF | ★★★★★ | ★★★☆☆ | **Très haute** |
| 3 | Streaming model loading | ★★★★☆ | ★★★☆☆ | **Haute** |
| 1 | Migration async (FastAPI) | ★★★★☆ | ★★★★☆ | Haute |
| 6 | KV cache offloading | ★★★★☆ | ★★★☆☆ | Haute |
| 9 | Continuous batching | ★★★★☆ | ★★★★☆ | Haute |
| 2 | Connection pool Tokio | ★★★☆☆ | ★★☆☆☆ | Moyenne |
| 8 | Plugin system | ★★★☆☆ | ★★☆☆☆ | Moyenne |
| 7 | Dashboard WebSocket | ★★☆☆☆ | ★★☆☆☆ | Moyenne |
| 16 | LoRA hot-swap | ★★★★☆ | ★★★☆☆ | Moyenne |
| 13 | MLX backend | ★★★☆☆ | ★★★☆☆ | Moyenne |
| 11 | Auto-scaling | ★★★☆☆ | ★★★★☆ | Basse |
| 10 | XDP complet | ★★☆☆☆ | ★★★★★ | Basse |
| 14 | Multi-modèle | ★★★☆☆ | ★★★★☆ | Basse |
| 15 | Fédération | ★★☆☆☆ | ★★★★★ | Exploratoire |
| 12 | Export ONNX/CoreML | ★★☆☆☆ | ★★★☆☆ | Exploratoire |
