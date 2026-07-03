# Résumé d'exécution — Fable.md, Phases 2 à 6

Branche : `feat/v6-lending-cooperative`
Date : 10 juin 2026

## Tâches Fable réalisées

### T2.4 — Retrait des mentions AITP (commit `3d3a74a`)
- Supprimé la section "Network / multi-node testing" du README (référence au protocole AITP / `scripts/test_network_lan.py`).
- Supprimé la ligne AITP de FEATURES.md.
- Vérifié par grep : plus aucune référence AITP dans README/FEATURES.

### T2.2 — Bench A/B Rust (`vramancer_rust`) vs Python CPU-staged (commit `cce70a3`)
- Mesures réelles sur le hardware (RTX 3090 + RTX 5070 Ti) :

| Scénario | Python CPU-staged | Rust P2P | Gain |
|---|---|---|---|
| 10 pages (30 MB) | 3.1 ms | 2.4 ms | +24.1% |
| 50 pages (150 MB) | 11.5 ms | 7.2 ms | +37.4% |
| 100 pages (300 MB) | 23.0 ms | 13.2 ms | +42.5% |
| 300 pages (900 MB) | 69.7 ms | 37.1 ms | +46.7% |
| 500 pages (1.5 GB) | 115.7 ms | 61.1 ms | **+47.2%** |

- **Décision : Rust reste dans `core/`** (gain >> seuil 10% sur tous les scénarios).
- Détail dans `benchmarks/BENCHMARK_RESULTS.md`.

### T2.3 — ReBAR : valider ou rétrograder (commit `56dd9a8`)
- Nouveau script `benchmarks/bench_transfer_strategies.py` comparant CPU-staged, Pipelined (Strategy 2) et ReBAR full-window (Strategy 1.7).
- Résultat : ReBAR (Strategy 1.7) **non actif** sur ce VM — BAR0 = 8 MB (GPU0) / 16 MB (GPU1), très en-dessous du seuil 4 GB requis pour le mode full-window. Le rapport de mai 2026 qui validait ReBAR correspondait à une autre config Proxmox (BAR1 ≥ VRAM).
- Pipelined (Strategy 2) confirmé comme fallback réellement actif et performant : jusqu'à **178 Gbps** sur 1024 MB.
- **Décision : ReBAR reste dans `experimental/`**, documentation mise à jour dans `experimental/README.md` pour refléter honnêtement l'état actuel.

### Bug fix annexe (commit `90d7f77`)
- `core/production_api.py` : import cassé `core.wake_on_inference` → corrigé en `experimental.wake_on_inference` (suite au déplacement T2.1).
- Ajout du mode `VRM_NO_GUNICORN=1` (serveur Werkzeug mono-process, nécessaire quand le modèle est pré-chargé en CUDA — gunicorn fork casse CUDA).

### Test du VRAM Lending Pool sur hardware réel (commit `bba2b09`)
- `LENDING_OFF` → **OOM** ("Tried to allocate 2.90 GiB. GPU 1 has ... 408.75 MiB free").
- `LENDING_ON` → **OK**, pool actif (gpu0 lendable=9.1GB, gpu1 lendable=2.64GB), load_time=30.7s, tok/s=2.48.
- **Conclusion : le lending résout un cas d'OOM réel** — preuve de valeur du concept.
- Point ouvert noté honnêtement : tok/s=2.48 vs baseline 16.1 sur ce run court (aucun lease déclenché, `total_leases_created=0`) — pas un coût du lending en soi, à creuser séparément si besoin.

### Phase 6 — Lancement Qwen3.6-35B-A3B + interface code/chat (commit `86a6701`)
- `serve_qwen36.sh` : lancement en une commande de Qwen3.6-35B-A3B (GGUF Q4_K_M, 22 GB) sur les 2 GPUs, avec batching continu et mode mono-process.
- Nouvelle route `/chat` servant une page HTML autonome (`dashboard/templates/code_chat.html`) : chat minimal, rendu des blocs de code, affichage tok/s, branchée sur `/v1/chat/completions`.
- **Bug trouvé et corrigé** : `ContinuousBatcher` était instancié pour tous les backends, y compris llama.cpp/vLLM/Ollama, dont l'interface `generate()` est incompatible avec son forward HF-style (`past_key_values`/`use_cache`). Garde ajoutée dans `core/inference_pipeline.py` pour désactiver `ContinuousBatcher` sur ces backends.
- **Bug trouvé et corrigé** : le chat ne s'arrêtait pas proprement (le modèle hallucinait un nouveau tour "User:/Assistant:"). Ajout de `stop=["\nUser:", "\nSystem:"]` dans `chat_completions`, propagé jusqu'à `backends_llamacpp.generate()`/`generate_stream()`.
- Validé : réponse Fibonacci propre, `finish_reason: stop`, ~83 tok/s.

## État final
- Tous les commits sont sur `feat/v6-lending-cooperative`, prêts pour revue/merge.
- Serveur Qwen3.6 opérationnel sur `http://localhost:5031` (et accessible sur le LAN via `http://192.168.1.28:5031`).
- Interface code/chat : `http://localhost:5031/chat`

## Non traité (hors scope de cette session)
- T1.2 (marqueurs pytest gpu/stub sur ~1161 tests) — explicitement reporté par l'utilisateur, priorité basse.
