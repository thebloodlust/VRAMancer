# Audit — core/orchestrator/ (5 modules)

## orchestrator/__init__.py (~5 LOC) — ✅ Bon
Re-exports PlacementEngine et BlockOrchestrator.

## orchestrator/adaptive_routing.py (~80 LOC) — 🔴 Non fonctionnel
Stubs avec `print()` debug. Pas de vraie logique de routage.
- 🔴 Weight compression `zlib` sur objets Python — échouera sur tenseurs réels
- 🔴 Threads sans daemon ni cleanup
- 🔴 **Code mort** — le routage réel est dans block_router.py

## orchestrator/block_orchestrator.py (~200 LOC) — ⚠️ Mixte
Orchestration placement mémoire hiérarchique (VRAM/DRAM/NVMe/network).
- 🔴 `load_block_from_disk()` et `save_block_to_disk()` lèvent NotImplementedError
- 🟡 Parsing status GPU fragile (string avec '% VRAM')
- 🟡 Potentiel boucle infinie si nœuds distants offline

## orchestrator/heterogeneous_manager.py (~400 LOC) — ✅ Bon
Gestion clusters hétérogènes (CUDA, ROCm, MPS, CPU, Edge/IoT).
- ✅ Scoring de ressources complet
- ⚠️ Patterns architecture GPU hardcodés
- ⚠️ Détection Edge par heuristiques simples (8GB RAM)

## orchestrator/placement_engine.py (~450 LOC) — ⚠️ Mixte
Placement production avec stratégies pluggables (profiled/VRAM/balanced).
- 🔴 `global_connectome` non défini — crash si stratégie neuroplasticité utilisée
- 🟡 Cache profils GPU jamais invalidé
- 🟡 Premier `place_model()` lent (benchmark tous GPU)
