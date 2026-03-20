# Audit — Modules core/ secondaires

## benchmark.py (~600+ LOC) — ⚠️ Bugs
Benchmarking d'inférence standardisé (tokps, TTFT, ITL).
- 🟡 Mode concurrent racy (pipeline partagé sans locks)
- 🟡 Calcul TTFT incorrect
- 🟡 P95/P99 off-by-one possible

## block_metadata.py (~50 LOC) — ✅ Placeholder
Métadonnées statiques + dynamiques pour blocs transformer.
- ⚠️ Estimations très inexactes ("attention: 800MB" pour tout modèle)

## continuous_batcher.py (~1000+ LOC) — 🔴 INCOMPLET
Batching continu itératif. **FICHIER TRONQUÉ** — KV cache batching manquant.
- 🔴 **NE PAS UTILISER** — implémentation coupée à la ligne ~520

## cross_vendor_bridge.py (~1200+ LOC) — 🔴 INCOMPLET
Transferts GPU-to-GPU AMD ↔ NVIDIA. **FICHIER TRONQUÉ** — transports manquants.
- ✅ Architecture excellente (DMA-BUF, ReBAR, pipeline async)
- 🔴 **NE PAS UTILISER** — implémentation incomplète

## gpu_fault_tolerance.py (~700+ LOC) — ✅ Solide (A-)
Détection/isolation pannes GPU, migration, auto-recovery.
- ⚠️ Recovery probe faible (torch.zeros 16x16)
- ⚠️ Pas de backoff après échecs répétés

## gpu_interface.py (~50 LOC) — ✅ Utilitaire léger
Énumération GPU et exécution sur GPU secondaires.

## health.py (~300+ LOC) — ✅ Bon
Diagnostics santé production (GPU, mémoire, température, fault state).

## hetero_config.py (~600+ LOC) — ✅ Excellent (A+)
Auto-détection clusters hétérogènes, base de données ~30 GPU (Blackwell, RDNA3).
- ⚠️ Fuzzy matching GPU fragile

## model_hub.py (~50 LOC) — 🔴 STUB
Recherche HuggingFace Hub par heuristiques filename naïves.

## persistence.py (~50 LOC) — ✅ Minimal
Persistance SQLite légère pour workflows.
- ⚠️ Pas de versioning de schéma

## speculative_decoding.py (~80 LOC) — 🔴 INCOMPLET
Décodage spéculatif swarm — **boucle principale non implémentée**.

## swarm_ledger.py (~100 LOC) — ⚠️ Basic
Ledger économique P2P (API keys, crédits VRAM).
- 🟡 Pas de log d'audit, brute-force possible

## tokenizer.py (~30 LOC) — ✅ Bon
Wrapper léger AutoTokenizer avec cache mémoire.
- ⚠️ Cache illimité

## vram_lending.py (~800+ LOC) — 🔴 INCOMPLET
Pool de prêt VRAM spéculatif. **FICHIER TRONQUÉ** — `_select_lender()` manquant.

## wake_on_inference.py (~25 LOC) — ⚠️ Fonctionne mais insécurisé
WoL standard sans authentification. `except Exception: pass` avale les erreurs.
