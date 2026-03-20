# Audit — core/network/ (32 modules, ~3500 LOC)

## Résumé global
Couche réseau avec transport adaptatif multi-tiers : RDMA/GPUDirect → TCP zero-copy → UDP anycast. Discovery mDNS/ZeroConf et UDP broadcast. Maturité mixte (production et prototypes coexistent).

| Critère | Évaluation |
|---------|------------|
| **LOC total** | ~3500 |
| **Qualité globale** | ⚠️ Mixte |
| **Sécurité globale** | 🔴 Risque élevé |

---

## Modules production-ready
| Module | LOC | Grade | Description |
|--------|-----|-------|-------------|
| cluster_discovery.py | ~280 | A- | mDNS/ZeroConf + UDP broadcast, heartbeat, thread-safe |
| fibre_fastpath.py | ~400+ | A | RDMA/GPUDirect RDMA, zero-copy TCP fallback |
| llm_transport.py (VTP) | ~400+ | A | Protocole tenseur binaire GPU-natif 64B header |
| transport.py | ~150 | A | TCP avec TLS optionnel, length-prefixed |
| trust_ring.py | ~50 | B+ | Isolation swarm par tokens d'invitation |
| security.py | ~50 | B+ | Auth nœud via HMAC, wrapping TLS |
| connectome.py | ~200 | B+ | Monitoring adaptatif synaptique (EMA, heartbeat) |
| auto_repair.py | ~120 | B+ | Redispatch auto de tâches vers nœuds sains |
| speculative_decoding.py | ~150 | B+ | Décodage spéculatif distribué |

## Modules à risque
| Module | LOC | Grade | Problème principal |
|--------|-----|-------|-------------------|
| cluster_master.py | ~60 | D | Prototype, aucune auth, JSON UDP non validé |
| actions.py | ~40 | F | Requêtes HTTP POST sans auth (reboot/failover) |
| remote_executor.py | ~30 | F | **pickle.loads() = RCE** |
| supervision_api.py | ~400+ | D | API Flask sans auth, heartbeat sans validation |
| swarm_inference.py | ~400+ | D | TCP accepte tenseurs de n'importe qui |
| transmission.py | ~200+ | D | Pickle comme fallback sérialisation |
| packet_builder.py | ~60 | D | SHA256 tronqué 64 bits, pas d'auth |
| aitp_protocol.py | ~80 | D | Magic bytes "VT" seule validation |
| aitp_sensing.py | ~100 | D | Attaque Sybil triviale (TFLOPS/VRAM spoofable) |
| aitp_network_raid.py | ~110 | C | eBPF non implémenté, pas de signatures shards |

## Modules stubs/incomplets
| Module | LOC | Status |
|--------|-----|--------|
| speculative_stream.py | ~50 | Stub — `guess_next_tokens()` hardcodé |
| neural_compression.py | ~80 | Simulation — quantification commentée |
| edge_iot.py | ~40 | RAM hardcodé à 1024, détection par string |
| network_trace.py | ~20 | Diagnostic optionnel (scapy) |

---

## Problèmes de sécurité critiques
1. 🔴 **Pas d'auth cluster** : UDP broadcast + mDNS acceptent n'importe quel nœud
2. 🔴 **Pickle RCE** : remote_executor.py + transmission.py (fallback)
3. 🔴 **Actions distantes non authentifiées** : actions.py POST arbitraire
4. 🔴 **Pas de vérification d'identité peer** : swarm_inference.py
5. 🔴 **WebGPU plaintext** : webgpu_node.py sans TLS
6. 🔴 **Checksums faibles** : packet_builder.py (SHA256 tronqué)

## Recommandations
1. Implémenter mutual TLS sur toute communication mesh
2. Remplacer pickle par MessagePack/Protobuf
3. Ajouter HMAC-SHA256 sur tous les types de paquets
4. Rate limiting + auth sur supervision_api.py
5. Désactiver AITP/swarm dans les déploiements cloud publics
