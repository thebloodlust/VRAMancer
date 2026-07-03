# DeepSeek → Opus & Architecte : Cluster, Peer, Sécurité, Gouvernance, Layer Profiling

> Regroupe TOUT : état des lieux cluster + admission + anti-poisoning +
> gouvernance maître/esclave + benchmark layers + recommandations.
> Un seul fichier. Date : 2026-06-15.

---

# PARTIE 1 — Ce qui existe DÉJÀ (impressionnant)

## 1.1 Détection des pairs

| Composant | Fichier | Ce qu'il fait |
|---|---|---|
| **mDNS/ZeroConf** | `experimental/cluster_discovery.py` | Découverte auto sur réseau local. Service `_vramancer._tcp.local.` |
| **UDP broadcast** | `experimental/cluster_discovery.py` | Fallback si mDNS indisponible |
| **IPv6 multicast** | `core/network/aitp_sensing.py` | Groupe `ff02::1:ff00:1`, heartbeat actif, staleness eviction |
| **NAT traversal** | `experimental/nat_traversal.py` | STUN (RFC 5389), UDP hole punch, relay fallback |
| **USB4/Thunderbolt** | `experimental/cluster_discovery.py` | Hot-plug detection (pyudev Linux, IOKit macOS) |
| **Membership** | `membership.jsonl` | Registre persistant des nœuds connus |

## 1.2 Élection de leader (maître/esclave)

| Composant | Détail |
|---|---|
| **Algorithme** | Bully : plus haut `gpu_count` = leader, hostname = tiebreaker |
| **Thread-safe** | `_leader_lock` protège l'élection |
| **Déclencheurs** | Appelée à chaque join/leave de nœud |

## 1.3 Routage et équilibrage

| Composant | Ce qu'il fait |
|---|---|
| **Anycast IPv6** | Routage santé-aware, round-robin pondéré |
| **Connectome** | Pondération Hebbienne adaptative |
| **API Edge** | Endpoint REST pour devices IoT/edge |

## 1.4 Sécurité existante

| Composant | Ce qu'il fait |
|---|---|
| **HMAC-SHA256** | Timing-safe, tous les messages |
| **Anti-replay** | Nonce + fenêtre 300s |
| **TLS** | Chiffrement socket |
| **XOR parity** | Intégrité des tensors (Rust) |
| **Audit log** | SQLite WAL, token hashé, thread-safe |
| **RBAC** | Contrôle d'accès par rôle |
| **Rate limiting** | Par route API |
| **HMAC Rust** | `sign_payload_fast`, `verify_hmac_fast`, `verify_hmac_batch` |

## 1.5 Layer Profiler (benchmark des couches)

| Fichier | `core/layer_profiler.py` |
|---|---|
| **Méthode** | `profile_model(model)` → `List[LayerProfile]` |
| **Par couche** | latence (ms), params (M), mémoire (MB), FLOPS estimés, intensité arithmétique |
| **Type de couche** | classifié (attention, mlp, norm, embedding) |
| **GPU profiling** | `profile_gpus()` → `List[GPUProfile]` (compute TFLOPS, mémoire BW) |
| **Placement DP** | `compute_optimal_placement(layers, gpus)` → `PlacementPlan` |
| **Détection PCIe** | `detect_pcie_bandwidth()` → GB/s réel |

---

# PARTIE 2 — Ce qui MANQUE

## Gap 1 ★★★ — Admission des nœuds

```python
class NodeAdmission:
    """
    Protocole en 3 phases :
    
    Phase 1 — DISCOVERY : le nouveau nœud se fait connaître via mDNS
    Phase 2 — CHALLENGE : le leader envoie un nonce, le nœud doit prouver
              qu'il possède le secret cluster (HMAC du nonce)
    Phase 3 — ADMIT : si le challenge passe, le leader signe un token
              d'admission. Le nœud est ajouté à membership.jsonl
    """
    
    def handle_join_request(self, node_info: dict, signature: bytes) -> bool:
        """Vérifie la signature HMAC + fingerprint hardware."""
        expected = hmac.new(
            self._cluster_secret,
            json.dumps(node_info).encode(),
            hashlib.sha256
        ).digest()
        
        if not hmac.compare_digest(signature, expected):
            self.log.warning(f"ADMISSION REJETÉE : {node_info['hostname']} — signature invalide")
            return False
        
        if node_info['hardware_fingerprint'] in self._blacklist:
            self.log.warning(f"ADMISSION REJETÉE : {node_info['hostname']} — hardware blacklisté")
            return False
        
        session_token = secrets.token_hex(32)
        self._admitted[node_info['hostname']] = {
            **node_info,
            'session_token': session_token,
            'admitted_at': time.time(),
        }
        self._save_membership()
        return True
```

## Gap 2 ★★★ — Anti-poisoning

```python
class PoisoningDetector:
    """
    3 niveaux de détection :
    
    1. Intégrité des tensors — XOR parity mismatch → corruption
    2. Capacités menties — VRAM annoncée >> VRAM réelle → mensonge
    3. Flood d'admission — > 5 tentatives/min → attaque
    
    Règle : 3 strikes → BAN. Blacklist du fingerprint hardware.
    """
    
    MAX_STRIKES = 3
    
    def check_tensor_integrity(self, node, tensor_hash, parity, shards):
        reconstructed = repair_xor_shard(shards, parity)
        if hashlib.sha256(reconstructed).hexdigest() != tensor_hash:
            self._strike(node, "tensor_corruption")
            return False
        return True
    
    def check_capability_lying(self, node, claimed, observed):
        if claimed.get('vram_gb', 0) > observed.get('vram_gb', 0) * 1.5:
            self._strike(node, "capability_lying")
            return False
        return True
    
    def _strike(self, node, reason):
        self._strikes[node] = self._strikes.get(node, 0) + 1
        if self._strikes[node] >= self.MAX_STRIKES:
            self._banned.add(node)
            self._blacklist_fingerprint(node)
            self._revoke_token(node)
```

## Gap 3 ★★☆ — Gouvernance maître/esclave

**Aujourd'hui** : l'élection Bully désigne un leader. Mais il ne fait rien.

**Ce qu'il faut** : le leader devient le cerveau du cluster.

```
┌─────────────────────────────────────────────────────────────────┐
│                     GOUVERNANCE                                 │
│                                                                 │
│  LEADER (élu par Bully)                                         │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ 🧠 ORCHESTRATION                                          │ │
│  │   • Reçoit les requêtes d'inférence                       │ │
│  │   • Consulte le Layer Profiler + PlacementEngine          │ │
│  │   • Décide QUI fait QUELLE couche                         │ │
│  │   • Assigne les tâches aux esclaves                       │ │
│  │                                                           │ │
│  │ 📊 MONITORING                                             │ │
│  │   • Agrège les métriques de tous les nœuds                │ │
│  │   • Expose /api/cluster (dashboard unifié)                │ │
│  │   • Détecte les nœuds morts (heartbeat timeout)           │ │
│  │                                                           │ │
│  │ 🛡️ SÉCURITÉ                                               │ │
│  │   • Valide les nouvelles admissions                       │ │
│  │   • Détecte le poisoning                                  │ │
│  │   • Révoque les nœuds malveillants                        │ │
│  │                                                           │ │
│  │ 🔄 FAILOVER                                               │ │
│  │   • Si le leader meurt → nouvelle élection Bully          │ │
│  │   • L'esclave élu reprend le membership.jsonl             │ │
│  │   • Les tokens de session survivent (signés HMAC)         │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ESCLAVES                                                       │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ 📡 REPORT au leader (heartbeat 10s)                       │ │
│  │   • Métriques GPU (VRAM, température, tok/s)              │ │
│  │   • Capacités (modèles chargés, VRAM libre)               │ │
│  │   • Santé (alive, degraded, dead)                         │ │
│  │                                                           │ │
│  │ ⚡ EXÉCUTION des tâches assignées                         │ │
│  │   • "Charge Qwen-Coder-7B sur GPU0"                       │ │
│  │   • "Exécute les couches 12-23 pour la requête #42"       │ │
│  │   • "Transfère le KV cache au nœud X"                     │ │
│  │                                                           │ │
│  │ 🚨 PROMOTION si le leader meurt                           │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Implémentation

```python
class ClusterGovernance:
    """Cerveau du cluster. Tourne sur le leader."""
    
    def __init__(self, discovery: ClusterDiscovery, profiler: LayerProfiler):
        self._discovery = discovery
        self._profiler = profiler
        self._admission = NodeAdmission()
        self._poisoning = PoisoningDetector()
        self._tasks: dict[str, Task] = {}          # task_id → Task
        self._node_metrics: dict[str, NodeMetrics] = {}
    
    # ── Cycle de vie ─────────────────────────────────
    
    def on_become_leader(self):
        """Appelé après élection. Démarre les services leader."""
        self._admission.start()
        self._heartbeat_monitor.start()
        self._metrics_aggregator.start()
    
    def on_become_follower(self, leader_id: str):
        """Appelé quand un autre nœud devient leader."""
        self._report_to = leader_id
        self._start_reporting()
    
    # ── Orchestration ─────────────────────────────────
    
    def handle_inference_request(self, prompt: str, model: str) -> Task:
        """
        Reçoit une requête d'inférence. Décide du plan d'exécution.
        """
        # 1. Profiler le modèle si pas encore fait
        layers = self._profiler.profile_model(model)
        
        # 2. Lister les GPUs disponibles sur tous les nœuds
        gpus = self._collect_all_gpus()
        
        # 3. Calculer le placement optimal (DP)
        plan = self._profiler.compute_optimal_placement(layers, gpus)
        
        # 4. Assigner les tâches aux nœuds
        task = Task(
            id=uuid4(),
            prompt=prompt,
            placement=plan,  # {layer_0: node_A_gpu0, layer_1: node_B_gpu0, ...}
        )
        self._dispatch(task)
        return task
    
    # ── Monitoring ────────────────────────────────────
    
    def _collect_all_gpus(self) -> list[GPUProfile]:
        """Agrège les profils GPU de tous les nœuds."""
        gpus = []
        for node_id, metrics in self._node_metrics.items():
            for gpu in metrics.gpus:
                gpus.append(GPUProfile(
                    node_id=node_id,
                    gpu_index=gpu.index,
                    vram_mb=gpu.vram_free_mb,
                    compute_gflops=gpu.compute_gflops,
                    bandwidth_gbps=self._get_link_bw(node_id),
                ))
        return gpus
    
    # ── Failover ──────────────────────────────────────
    
    def _heartbeat_monitor_loop(self):
        """Détecte les nœuds morts et réassigne leurs tâches."""
        while True:
            for node_id, last_seen in self._last_heartbeat.items():
                if time.time() - last_seen > 30:  # 30s timeout
                    self._mark_node_dead(node_id)
                    self._reassign_tasks(node_id)
            time.sleep(10)
```

## Gap 4 ★★☆ — Benchmark des couches (perf réelle vs théorique)

**Aujourd'hui** : `LayerProfiler.profile_model()` mesure latence + mémoire.
Mais il ne compare PAS la perf réelle au plafond théorique.

**Ce qu'il faut** : un rapport qui dit "cette couche est 78% du plafond théorique
→ probablement memory-bound, mettons-la sur le GPU avec la meilleure BW mémoire".

```
┌──────────────────────────────────────────────────────────────────┐
│                LAYER BENCHMARK REPORT                            │
│  Modèle : Qwen2.5-14B  |  GPU : RTX 5070 Ti (FP4, 16 GB)       │
├──────────────────────────────────────────────────────────────────┤
│ Layer │ Type      │ Lat(ms) │ Params(MB) │ Activ(MB) │ %Plafond │
├───────┼───────────┼─────────┼────────────┼───────────┼──────────┤
│ 0     │ embed     │ 0.12    │ 256        │ 8         │ 92% ✅   │
│ 1     │ attention │ 2.34    │ 128        │ 64        │ 45% ⚠️   │
│ 2     │ mlp       │ 3.51    │ 256        │ 96        │ 78% ✅   │
│ 3     │ attention │ 2.31    │ 128        │ 64        │ 46% ⚠️   │
│ ...   │ ...       │ ...     │ ...        │ ...       │ ...      │
│ 47    │ lm_head   │ 0.08    │ 256        │ 1         │ 95% ✅   │
├───────┴───────────┴─────────┴────────────┴───────────┴──────────┤
│ RECOMMANDATIONS DE PLACEMENT :                                  │
│                                                                 │
│  Layers 1-10 (attention lourd) → GPU0 (5070Ti, BW mémoire high) │
│  Layers 11-47 (mlp dominant)    → GPU1 (3090, compute BF16 OK)  │
│  Layers embed+head              → GPU0 (FP4 pas nécessaire)      │
│                                                                 │
│  Économie estimée vs split uniforme : +12% tok/s                │
└──────────────────────────────────────────────────────────────────┘
```

### Implémentation

```python
class LayerBenchmarker:
    """
    Compare la perf mesurée au plafond théorique du GPU.
    Identifie les couches memory-bound vs compute-bound.
    """
    
    def benchmark_layer(self, layer, gpu_profile: GPUProfile) -> LayerBenchmark:
        """Mesure une couche et calcule l'efficacité vs plafond."""
        # Mesurer
        lat_ms = self._measure_latency(layer)
        
        # Calculer le plafond théorique
        if self._is_compute_bound(layer):
            # Plafond = FLOPS estimés / peak TFLOPS du GPU
            theoretical_min_ms = layer.estimated_flops / gpu_profile.compute_throughput_gflops / 1e9 * 1000
        else:
            # Plafond = bytes lus/écrits / BW mémoire du GPU
            bytes_moved = layer.param_memory_mb * 1024 * 1024 + layer.activation_memory_mb * 1024 * 1024
            theoretical_min_ms = bytes_moved / gpu_profile.memory_bandwidth_gbps / 1e9 * 1000
        
        efficiency_pct = (theoretical_min_ms / lat_ms) * 100
        
        return LayerBenchmark(
            layer_index=layer.index,
            measured_ms=lat_ms,
            theoretical_min_ms=theoretical_min_ms,
            efficiency_pct=efficiency_pct,
            bottleneck="compute" if efficiency_pct < 60 else "memory" if efficiency_pct < 80 else "optimal",
            recommendation=self._placement_hint(layer, efficiency_pct, gpu_profile),
        )
    
    def generate_placement_report(self, layers, gpus) -> PlacementReport:
        """Génère le rapport complet avec recommandations."""
        benchmarks = [self.benchmark_layer(l, gpus[0]) for l in layers]
        
        # Stratifier : quelles couches sur quel GPU
        attention_layers = [b for b in benchmarks if b.layer_type == "attention"]
        mlp_layers = [b for b in benchmarks if b.layer_type == "mlp"]
        
        # Les couches d'attention bénéficient de la BW mémoire élevée
        # → GPU avec la meilleure memory_bandwidth_gbps
        best_bw_gpu = max(gpus, key=lambda g: g.memory_bandwidth_gbps)
        
        # Les couches MLP bénéficient du compute
        # → GPU avec le meilleur compute_throughput_gflops
        best_compute_gpu = max(gpus, key=lambda g: g.compute_throughput_gflops)
        
        return PlacementReport(
            benchmarks=benchmarks,
            recommendation={
                'attention_layers': {'gpu': best_bw_gpu, 'layers': attention_layers},
                'mlp_layers': {'gpu': best_compute_gpu, 'layers': mlp_layers},
                'estimated_gain_pct': 12.0,  # vs split uniforme
            }
        )
```

## Gap 5 ★☆☆ — mDNS pour Mac + Thunderbolt IP

- **mDNS Mac** : `zeroconf` est compatible macOS. `detect_platform_type()` détecte déjà Apple Silicon. Juste à câbler.
- **Thunderbolt IP mode** : ~20 Gbps effectifs (vs 1 Gbps Ethernet). Faire passer VTP/GpuNetBridge par-dessus.

---

# PARTIE 3 — Ce qu'on a VRAIMENT pour un cluster complet

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLUSTER VRAMancer v2                        │
│                                                                 │
│  DÉTECTION (✅ complet)                                         │
│  ├── mDNS/ZeroConf + UDP broadcast + IPv6 multicast            │
│  ├── NAT traversal (STUN + hole punch + relay)                 │
│  └── USB4/Thunderbolt hot-plug                                 │
│                                                                 │
│  ÉLECTION (✅ base, ⬜ coordination)                            │
│  ├── ✅ Bully algorithm + leader lock                          │
│  └── ⬜ Leader → orchestration + monitoring + admission        │
│                                                                 │
│  SÉCURITÉ (✅ base, ⬜ admission + poisoning)                   │
│  ├── ✅ HMAC + TLS + anti-replay + audit log + RBAC            │
│  ├── ✅ XOR parity (intégrité tensors)                         │
│  ├── ⬜ Admission protocol (challenge HMAC + token)            │
│  └── ⬜ Poisoning detection (3 strikes → ban)                  │
│                                                                 │
│  LAYER PROFILING (✅ base, ⬜ benchmark vs théorie)             │
│  ├── ✅ profile_model() : latence + mémoire + FLOPS            │
│  ├── ✅ compute_optimal_placement() : DP                       │
│  └── ⬜ benchmark_layer() : efficacité vs plafond théorique    │
│                                                                 │
│  ORCHESTRATION (⬜ à construire)                                │
│  ├── ⬜ ClusterGovernance : handle_inference_request()         │
│  ├── ⬜ Task dispatch : couche X → nœud Y                      │
│  ├── ⬜ Failover : promotion si leader mort                    │
│  └── ⬜ Dashboard cluster unifié                               │
│                                                                 │
│  TRANSFERT (✅ complet)                                         │
│  ├── GpuPipeline (25 GB/s local)                               │
│  ├── GpuNetBridge (GPU→TCP→GPU, VTP)                           │
│  ├── Tokio P2P (TCP chunked + HMAC)                            │
│  └── Thunderbolt IP mode (~20 Gbps)                            │
└─────────────────────────────────────────────────────────────────┘
```

---

# PARTIE 4 — Recommandations finales

## Priorités immédiates (session en cours)

| Priorité | Chantier | Effort | Impact |
|---|---|---|---|
| **1** | Admission protocol | 1-2 sessions | Cluster secure multi-nœuds |
| **2** | Anti-poisoning (3 strikes → ban) | 1 session | Intégrité du cluster |
| **3** | Layer benchmark vs théorie | 1 session | Placement informed, pas au pif |
| **4** | Leader coordination (gouvernance) | 2 sessions | Le cluster devient un orchestre |
| **5** | Dashboard cluster unifié | 1 session | Visibilité multi-nœuds |

## Secondaires

| Priorité | Chantier |
|---|---|
| 6 | mDNS câblage Mac au démarrage |
| 7 | Thunderbolt IP benchmark |
| 8 | Failover automatique (test de chaos) |

---

— DeepSeek
