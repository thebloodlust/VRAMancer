# VRAMancer â€” Plan de restructuration et de crÃ©dibilisation

> Document destinÃ© Ã  des agents de codage. Chaque tÃ¢che contient un objectif,
> des actions concrÃ¨tes et des critÃ¨res d'acceptation vÃ©rifiables.
> Ordre d'exÃ©cution recommandÃ© : Phase 0 â†’ 1 â†’ 2 â†’ 3 â†’ 4, puis 5 et 6.
> La Phase 5 (P2P) peut dÃ©marrer en parallÃ¨le dÃ¨s maintenant ; la Phase 6 (Qwen3.6)
> dÃ©pend de T6.1 mais pas des phases 0-4.
> RÃ¨gle d'or transverse : **ne rien afficher dans le README qui n'est pas
> validÃ© par un benchmark reproductible sur du matÃ©riel rÃ©el.**

---

## Phase 0 â€” HygiÃ¨ne immÃ©diate (quick wins, < 1 jour)

### T0.1 â€” Ajouter le fichier LICENSE
- CrÃ©er `LICENSE` Ã  la racine avec le texte MIT complet (copyright + annÃ©e + nom de l'auteur).
- **Acceptation** : GitHub affiche le badge "MIT license" sur la page du repo.

### T0.2 â€” Nettoyer la racine du dÃ©pÃ´t
- CrÃ©er `benchmarks/results/` et y dÃ©placer tous les `bench_*.json` et `bench_*.txt` de la racine.
- Supprimer ou dÃ©placer les fichiers orphelins : `bug`, `mac`, `mac_mlx`, `futur.md`, `_test_kernel.py`, `test_results.txt`, `download_awq.py` (â†’ `scripts/` si encore utile).
- DÃ©placer `mac_worker.py` vers `scripts/` ou `core/` selon son usage rÃ©el.
- Mettre Ã  jour tous les chemins rÃ©fÃ©rencÃ©s (scripts de bench, docs, CI).
- **Acceptation** : la racine ne contient plus que : dossiers de code, fichiers de config standards (`pyproject.toml`, `Makefile`, `Dockerfile`, `docker-compose.yml`, `.gitignore`, `pytest.ini`), docs Markdown principaux, `LICENSE`. `pytest` et les scripts de bench passent toujours.

### T0.3 â€” Consolider les requirements
- Fusionner `requirements.txt` / `requirements-lite.txt` / `requirements-full.txt` / `requirements-windows.txt` en extras dans `pyproject.toml` : `pip install vramancer[full]`, `[cuda]`, `[mac]`.
- Conserver un seul `requirements.txt` minimal pointant vers `pyproject.toml`, ou le supprimer.
- **Acceptation** : `pip install -e .` et `pip install -e .[full]` fonctionnent ; les docs d'install n'Ã©voquent plus 4 fichiers requirements.

---

## Phase 1 â€” Source unique de vÃ©ritÃ© pour les chiffres

### T1.1 â€” Synchroniser les benchmarks entre documents
IncohÃ©rences connues Ã  corriger :
- README : "6.0 tok/s" pour 14B 2-GPU â†’ le benchmark dÃ©taillÃ© donne **16.1 tok/s** (rÃ©sultat le plus rÃ©cent). Choisir le chiffre le plus rÃ©cent et reproductible.
- README + FEATURES.md : "NF4 75% plus rapide" â†’ obsolÃ¨te. Le benchmark rÃ©cent montre NF4 single-GPU = 10.5 tok/s vs BF16 2-GPU = 16.1 tok/s (NF4 est plus lent mais âˆ’70% VRAM). Reformuler en compromis vitesse/VRAM honnÃªte.
- Compteurs de tests divergents (957 / 901 / 853) â†’ un seul chiffre, mis Ã  jour automatiquement ou supprimÃ© des docs statiques.
- **RÃ¨gle** : `benchmarks/BENCHMARK_RESULTS.md` est la source unique. README et FEATURES.md ne contiennent que des extraits avec lien, jamais de chiffres divergents.
- **Acceptation** : `grep -rn "6.0 tok" --include="*.md"` et `grep -rn "75%" --include="*.md"` ne retournent plus de chiffre contredisant BENCHMARK_RESULTS.md.

### T1.2 â€” Distinguer tests stub vs tests GPU
- Marquer les tests avec `@pytest.mark.gpu` / `@pytest.mark.stub`.
- Documenter : "X tests unitaires (sans GPU, stubs) + Y tests d'intÃ©gration GPU".
- Ne plus prÃ©senter le total stub comme preuve de couverture fonctionnelle.
- **Acceptation** : `pytest -m stub` et `pytest -m gpu` sont exÃ©cutables sÃ©parÃ©ment ; README reflÃ¨te la distinction.

---

## Phase 2 â€” Trancher le pÃ©rimÃ¨tre (la tÃ¢che la plus importante)

### T2.1 â€” DÃ©finir le core et dÃ©placer le reste
**Core (reste dans `core/` et le README)** â€” uniquement ce qui est prouvÃ© par benchmark :
- Auto-dÃ©tection GPU + split VRAM-proportionnel (`model_splitter.py`)
- Pipeline d'infÃ©rence multi-GPU hÃ©tÃ©rogÃ¨ne + CPU offload
- Backends : HuggingFace, llama.cpp/GGUF, vLLM, Ollama
- Quantization : NF4, INT8, NVFP4 + DirectFP4 bypass
- TurboEngine (boucle compilÃ©e, +34-47% prouvÃ©)
- TurboQuant KV compression (version zero-overhead)
- Kernel CUDA PagedAttention (8.8x prouvÃ© Ã  contexte court)
- CLI `vramancer run/serve/status/...` + API OpenAI-compatible
- Monitoring de base (GPUMonitor, health, mÃ©triques Prometheus)

**Experimental (dÃ©placer dans `experimental/` avec un README d'avertissement)** :
- AITP protocol + FEC Reed-Solomon (`aitp_protocol.py`, `aitp_fec.py`)
- NAT traversal / STUN (`nat_traversal.py`)
- Wake-on-Inference (`wake_on_inference.py`)
- Cross-vendor bridge AMDâ†”NVIDIA (`cross_vendor_bridge.py`) â€” non testable sans GPU AMD
- RDMA / GPUDirect (`fibre_fastpath.py`) â€” non testable sans matÃ©riel IB/RoCE
- VRAM lending (`vram_lending.py`)
- Hierarchical memory 6 niveaux (`hierarchical_memory.py`) â€” sauf si benchmarkÃ©
- Cluster discovery / Bully election (`cluster_discovery.py`) â€” sauf si le multi-nÅ“ud devient un axe assumÃ©
- StratÃ©gie ReBAR (voir T2.3)

**DÃ©cision Rust (`rust_core/`)** : voir T2.2.

- CrÃ©er `experimental/README.md` : Â« Modules non validÃ©s sur matÃ©riel rÃ©el. APIs instables. Aucune garantie. Contributions de validation bienvenues. Â»
- Mettre Ã  jour tous les imports ; les modules expÃ©rimentaux doivent rester importables mais derriÃ¨re un flag `VRM_EXPERIMENTAL=1`.
- **Acceptation** : la suite de tests passe ; le README principal ne mentionne plus aucun module expÃ©rimental ; importer un module expÃ©rimental sans le flag lÃ¨ve un warning explicite.

### T2.2 â€” Statuer sur l'extension Rust par un benchmark A/B
- Ã‰crire `benchmarks/bench_rust_vs_python.py` : mÃªme transfert inter-GPU (taille d'activations rÃ©aliste, ex. 14B), chemin Rust GpuPipeline vs chemin Python CPU-staged, mesure tok/s end-to-end.
- Si gain end-to-end > 10 % : garder Rust dans le core et publier le chiffre.
- Sinon : dÃ©placer `rust_core/` dans `experimental/` et retirer du README.
- **Acceptation** : le rÃ©sultat du bench est dans BENCHMARK_RESULTS.md et la dÃ©cision (core ou experimental) est appliquÃ©e.

### T2.3 â€” ReBAR : valider ou rÃ©trograder
- La stratÃ©gie ReBAR n'a jamais pu Ãªtre testÃ©e (Proxmox/IOMMU bloque le P2P). C'est pourtant la brique la plus originale du projet.
- Action : prÃ©parer `benchmarks/bench_transfer_strategies.py` comparant Strategy 1 (P2P), 1.7 (ReBAR), 2 (CPU-pipelined), 4 (CPU-staged) â€” exÃ©cutable le jour oÃ¹ un accÃ¨s bare-metal est disponible.
- Tant que non validÃ© : `experimental/`, avec une note Â« designed, awaiting bare-metal validation Â».
- Si validÃ© avec gain : promouvoir en core et en faire un argument diffÃ©renciant majeur du README (quasi personne n'exploite ReBAR pour Ã§a).
- **Acceptation** : le script de bench existe et tourne (au moins en mode dÃ©gradÃ© sur la config actuelle) ; le statut du module est cohÃ©rent avec les rÃ©sultats.

### T2.4 â€” AITP : geler
- DÃ©placer dans `experimental/` (cf. T2.1) ou dans un repo sÃ©parÃ© `thebloodlust/aitp`.
- Justification : sur Ethernet grand public, la latence/bande passante inter-nÅ“uds domine tellement que le protocole custom n'apporte pas d'avantage mesurable vs TCP/NCCL ; c'est un projet de recherche, pas une feature produit.
- Ne plus le mentionner dans le README principal.
- **Acceptation** : aucune rÃ©fÃ©rence AITP dans README/FEATURES principaux.

---

## Phase 3 â€” README et positionnement

### T3.1 â€” RÃ©Ã©crire le README autour de la preuve
Structure cible :
1. Pitch une phrase + la commande `vramancer run Qwen/Qwen2.5-14B-Instruct`
2. **Le tableau de preuve** (OOM/OOM/16.1 tok/s) en haut, chiffres synchronisÃ©s avec T1.1
3. GIF/asciinema de dÃ©mo (voir T3.3)
4. Install (3 plateformes, condensÃ©)
5. **Section "Quand utiliser VRAMancer ?"** (voir T3.2)
6. Usage, backends, config
7. Lien vers BENCHMARK_RESULTS.md, architecture, experimental/
- Supprimer ou fusionner `README_FACILE.md` (un seul README, Ã©ventuellement une section "DÃ©marrage simple").
- Choisir une langue principale (l'anglais pour l'audience r/LocalLLaMA ; garder Ã©ventuellement `docs/README.fr.md`).
- **Acceptation** : README < 250 lignes, zÃ©ro feature non prouvÃ©e, zÃ©ro chiffre divergent.

### T3.2 â€” Section comparative honnÃªte
Tableau Â« VRAMancer vs alternatives Â» :
| Besoin | Outil recommandÃ© |
|---|---|
| ModÃ¨le GGUF, 1+ GPU | llama.cpp (que VRAMancer utilise comme backend) |
| Serving haut dÃ©bit, GPUs homogÃ¨nes | vLLM |
| Split auto HF basique | accelerate `device_map="auto"` |
| **ModÃ¨le HF trop gros, GPUs hÃ©tÃ©rogÃ¨nes, une commande, auto-sÃ©lection backend** | **VRAMancer** |
- Assumer que la valeur = UX une-commande + orchestration intelligente + optimisations propres (TurboEngine, TurboQuant, DirectFP4), pas la rÃ©invention des backends.
- **Acceptation** : la section existe et ne survend rien (chaque ligne dÃ©fendable face Ã  un lecteur expert).

### T3.3 â€” DÃ©mo visuelle
- Enregistrer un asciinema/GIF : `vramancer run Qwen2.5-14B` â†’ dÃ©tection des 2 GPUs â†’ split affichÃ© â†’ gÃ©nÃ©ration en live.
- L'intÃ©grer en haut du README.
- **Acceptation** : le GIF se charge sur la page GitHub et montre le parcours complet en < 30 s.

### T3.4 â€” Publication PyPI
- VÃ©rifier `pyproject.toml` (metadata, classifiers, entry point `vramancer`).
- Publier sur TestPyPI puis PyPI : `pip install vramancer`.
- Workflow GitHub Actions de release sur tag.
- **Acceptation** : `pip install vramancer && vramancer status` fonctionne sur une machine vierge.

---

## Phase 4 â€” CI, qualitÃ©, communautÃ©

### T4.1 â€” CI lisible
- GitHub Actions : lint (ruff ou flake8) + tests stub sur push/PR, matrix Python 3.10-3.12.
- Badges CI + PyPI + licence dans le README.
- Traiter les 3 PRs ouvertes (merger ou fermer).
- **Acceptation** : badge vert sur main, zÃ©ro PR dormante.

### T4.2 â€” Documentation d'architecture Ã  jour
- VÃ©rifier que `docs/architecture.md` reflÃ¨te le pÃ©rimÃ¨tre post-Phase 2 (sans les modules expÃ©rimentaux dans le schÃ©ma principal).
- Ajouter un diagramme du flux : CLI â†’ backend selection â†’ splitter â†’ transfer strategies â†’ inference.
- **Acceptation** : un nouveau contributeur comprend le flux en lisant un seul document.

### T4.3 â€” Lancement communautaire (aprÃ¨s Phases 0-3 uniquement)
- Post r/LocalLLaMA. Accroche : Â« RTX 3090 + RTX 5070 Ti = Qwen-14B bf16 qui OOM sur chaque carte seule â€” une commande Â». Public idÃ©al (beaucoup de configs GPU dÃ©pareillÃ©es).
- PrÃ©parer les rÃ©ponses aux objections prÃ©visibles : Â« llama.cpp fait dÃ©jÃ  Ã§a Â» (â†’ tableau T3.2), Â« pourquoi pas exo Â» (â†’ focus single-node multi-GPU vs multi-machine).
- Ã‰ventuellement Show HN ensuite.
- **Acceptation** : post publiÃ© avec lien vers un repo propre ; les questions techniques trouvent rÃ©ponse dans le README ou les benchmarks.

---

## Annexe â€” Verdict sur les briques exotiques (pour mÃ©moire)

| Brique | Statut recommandÃ© | Justification |
|---|---|---|
| Kernel CUDA PagedAttention | **Core** | ProuvÃ© (8.8x @ctx64), limites documentÃ©es |
| ReBAR DMA | Experimental â†’ core si validÃ© bare-metal | Brique la plus originale du projet, mais jamais testÃ©e (IOMMU bloque P2P en VM) |
| Rust (Tokio + CUDA FFI) | Bench A/B dÃ©cisif (T2.2) | Les benchs existants montrent que le goulot est la bande passante, pas Python ; coÃ»t de maintenance Ã©levÃ© |
| AITP + FEC | Experimental / repo sÃ©parÃ© | RÃ©invente NCCL/QUIC ; gain non mesurable sur Ethernet grand public ; signal "vaporware" pour les lecteurs |
| Cross-vendor AMDâ†”NVIDIA | Experimental | Non testable sans GPU AMD |
| RDMA / GPUDirect | Experimental | Non testable sans matÃ©riel IB/RoCE |
| NAT traversal, WoL, Bully election | Experimental | Hors du problÃ¨me core (infÃ©rence single-node multi-GPU) |

## Phase 5 â€” Validation et exploitation du P2P (NOUVEAU â€” dÃ©bloquÃ© par la conf Proxmox)

> Contexte : le P2P a Ã©tÃ© activÃ© dans Proxmox (modification de conf) et `nvidia-smi`
> semble le confirmer. Une mesure prÃ©liminaire indiquerait ~20 GB/s (Ã  re-vÃ©rifier â€”
> chiffre rapportÃ© de mÃ©moire, non documentÃ©). Baseline CPU-staged : ~11 GB/s.
> AUCUN chiffre P2P ne doit entrer dans les docs avant validation complÃ¨te T5.1-T5.3.

### T5.1 â€” VÃ©rification de la dÃ©claration driver
- ExÃ©cuter et archiver dans `benchmarks/results/p2p/` :
  ```bash
  nvidia-smi topo -m > benchmarks/results/p2p/topo.txt
  nvidia-smi topo -p2p rw >> benchmarks/results/p2p/topo.txt
  python -c "import torch; print(torch.cuda.can_device_access_peer(0,1), torch.cuda.can_device_access_peer(1,0))"
  ```
- Documenter la modification Proxmox exacte (fichier de conf, paramÃ¨tre, valeur) dans `docs/proxmox_p2p.md` â€” beaucoup d'utilisateurs homelab cherchent cette info, c'est un aimant Ã  trafic.
- **Acceptation** : `docs/proxmox_p2p.md` existe avec la conf reproductible ; les sorties topo sont archivÃ©es.

### T5.2 â€” Test de CORRECTION des donnÃ©es (bloquant, avant tout benchmark)
- Compiler et exÃ©cuter les CUDA samples `simpleP2P` (vÃ©rification d'intÃ©gritÃ©) et `p2pBandwidthLatencyTest` (bande passante rÃ©elle par paire).
- Si la conf Proxmox utilise un override ACS : risque de corruption silencieuse des Ã©critures DMA. `simpleP2P` DOIT afficher la vÃ©rification OK.
- Test applicatif complÃ©mentaire : gÃ©nÃ©rer 256 tokens en greedy decoding sur le 14B avec P2P actif, puis avec CPU-staged forcÃ© â€” les sorties doivent Ãªtre **identiques token pour token**.
- **Acceptation** : simpleP2P passe ; les deux gÃ©nÃ©rations greedy sont identiques (script de diff archivÃ©). En cas d'Ã©chec : dÃ©sactiver le P2P, documenter, et NE PAS poursuivre T5.3-T5.5.

### T5.3 â€” Mesurer la bande passante rÃ©elle et confirmer le ~20 GB/s
- Archiver la sortie de `p2pBandwidthLatencyTest` dans `benchmarks/results/p2p/`.
- Confirmer ou infirmer le ~20 GB/s rapportÃ©. Grille de lecture : ~11-14 GB/s = gain marginal vs CPU-staged ; â‰¥18 GB/s = gain significatif attendu sur le multi-GPU.
- **Acceptation** : le chiffre mesurÃ© (pas estimÃ©) figure dans BENCHMARK_RESULTS.md avec la mÃ©thodologie.

### T5.4 â€” Re-benchmark end-to-end avec P2P
- Relancer `bench_heterogeneous.py` sur Qwen2.5-14B 2-GPU : comparer P2P vs CPU-staged (baseline 16.1 tok/s). HypothÃ¨se Ã  valider : +10-30%.
- Relancer le bench bi-GPU 7B pour mesurer la rÃ©duction de la pÃ©nalitÃ© de split.
- Mettre Ã  jour BENCHMARK_RESULTS.md avec une section "P2P enabled (Proxmox)" et propager les nouveaux chiffres (rÃ¨gle T1.1).
- VÃ©rifier que la chaÃ®ne de stratÃ©gies de transfert sÃ©lectionne bien Strategy 1 (CUDA P2P) automatiquement maintenant que P2P est disponible â€” sinon corriger la dÃ©tection.
- **Acceptation** : tableau avant/aprÃ¨s P2P dans BENCHMARK_RESULTS.md ; la Strategy 1 est auto-sÃ©lectionnÃ©e (vÃ©rifiable dans les logs).

### T5.5 â€” Tester les stratÃ©gies dÃ©bloquÃ©es (ReBAR, Rust DtoD)
- Le P2P dÃ©bloquÃ© rend potentiellement testables : Strategy 1.7 (ReBAR) et Strategy 1.5 (Rust DtoD). ExÃ©cuter `bench_transfer_strategies.py` (cf. T2.3) sur les 4+ stratÃ©gies.
- Si ReBAR montre un gain mesurable : promotion experimental â†’ core + section README dÃ©diÃ©e (argument diffÃ©renciant quasi unique).
- Ce bench sert aussi d'arbitre pour la dÃ©cision Rust (T2.2).
- **Acceptation** : tableau comparatif des stratÃ©gies de transfert dans BENCHMARK_RESULTS.md ; dÃ©cisions ReBAR et Rust actÃ©es en consÃ©quence.

---

## Phase 6 â€” Stack de codage local : Qwen3.6 servi par VRAMancer (NOUVEAU)

> Objectif utilisateur : remplacer une partie des appels API payants par un modÃ¨le
> local lanÃ§able en une commande, branchÃ© sur des agents de codage (Claude Code via
> routeur, Aider, Cline, Qwen Code, etc.) via l'endpoint OpenAI-compatible existant.
> MatÃ©riel : RTX 3090 (24 GB) + RTX 5070 Ti (16 GB) = 40 GB VRAM, P2P actif (si T5.2 OK).

### T6.1 â€” Choisir et valider le modÃ¨le
- Cible principale : **Qwen3.6-35B-A3B** (MoE, 3B actifs â†’ dÃ©codage rapide, qualitÃ© coding agentique, 256K contexte). En GGUF quantifiÃ© il pÃ¨se ~17-24 GB selon le quant.
- Alternative : **Qwen3.6-27B** (dense, ~17 GB en quant 4-bit) si le MoE pose problÃ¨me cÃ´tÃ© llama.cpp/transformers.
- StratÃ©gie de placement Ã  benchmarker (3 variantes) :
  1. GGUF Q4_K_M sur la 3090 seule via backend llama.cpp (attendu : le plus rapide, cf. 106.8 tok/s sur 7B â€” un MoE 3B actifs devrait Ãªtre dans le mÃªme ordre de grandeur)
  2. GGUF avec tensor_split 3090+5070 Ti pour libÃ©rer de la marge KV cache (contexte long de coding agent : 32K-128K tokens, le KV cache devient le vrai consommateur)
  3. BF16/NF4 split VRAMancer HF (probablement plus lent, mais Ã  mesurer pour le README)
- **Acceptation** : tableau comparatif des 3 variantes (tok/s, TTFT, VRAM, contexte max supportÃ©) dans BENCHMARK_RESULTS.md ; une variante recommandÃ©e dÃ©signÃ©e.

### T6.2 â€” Commande de lancement une-ligne
- VÃ©rifier/corriger `vramancer run` et `vramancer serve` avec le repo GGUF Qwen3.6 (auto-download HF Hub, auto-sÃ©lection backend llama.cpp).
- Cible : `vramancer serve --model <repo-gguf-qwen3.6> --port 5030 --ctx 65536` fonctionne sans config supplÃ©mentaire.
- GÃ©rer le template de chat Qwen3 (reasoning/thinking blocks) : s'assurer que le parsing du format de sortie (balises de raisonnement) est correct dans l'API.
- **Acceptation** : la commande dÃ©marre, `curl /v1/chat/completions` rÃ©pond correctement avec et sans bloc de raisonnement.

### T6.3 â€” CompatibilitÃ© agents de codage (le point critique)
- Les agents de codage exigent du **function calling / tool use** fiable sur l'endpoint OpenAI-compatible. VÃ©rifier que l'API VRAMancer supporte : `tools`, `tool_choice`, rÃ©ponses `tool_calls`, streaming SSE.
- Qwen3.6 utilise le parser de tool-calls format qwen3_coder (cf. doc vLLM/SGLang : `--tool-call-parser qwen3_coder`). ImplÃ©menter l'Ã©quivalent dans `production_api.py` ou documenter le passage par vLLM backend pour ce cas.
- Tester end-to-end avec au moins deux clients rÃ©els : Aider (`aider --openai-api-base http://localhost:5030/v1`) et Cline/Continue ou Qwen Code. Documenter la config de chaque client dans `docs/coding_agents.md`.
- Option : documenter le routage de Claude Code vers l'endpoint local via un routeur OpenAI-compatible (pour les tÃ¢ches simples, en gardant l'API cloud pour les tÃ¢ches complexes â€” usage hybride Ã©conome).
- **Acceptation** : une session Aider complÃ¨te (Ã©dition multi-fichiers avec tool calls) aboutit contre le serveur local ; `docs/coding_agents.md` couvre â‰¥2 clients avec configs copiables.

### T6.4 â€” Contexte long et KV cache
- Les agents de codage envoient des contextes de 30-120K tokens. Mesurer : VRAM du KV cache Ã  32K/64K/128K, TTFT (prefill) et tok/s decode Ã  ces tailles.
- C'est ici que TurboQuant KV (version zero-overhead) a une vraie utilitÃ© produit : âˆ’13.8% VRAM = plus de contexte. Benchmarker avec/sans sur un prompt de 64K.
- DÃ©finir et documenter le contexte max recommandÃ© par variante de placement (T6.1).
- **Acceptation** : tableau contexte/VRAM/TTFT/tok-s dans BENCHMARK_RESULTS.md ; recommandation chiffrÃ©e dans docs/coding_agents.md.

### T6.5 â€” Cas d'usage vitrine pour le README
- Si T6.1-T6.4 aboutissent, ajouter au README une section Â« Local coding assistant Â» : 2 commandes (serve + config client) pour avoir un agent de codage local gratuit sur GPUs hÃ©tÃ©rogÃ¨nes.
- C'est un cas d'usage concret qui rÃ©pond Ã  Â« pourquoi VRAMancer et pas juste llama.cpp ? Â» : auto-dÃ©tection, split intelligent, API prÃªte pour agents, une commande.
- **Acceptation** : section README â‰¤ 30 lignes, testÃ©e en suivant les instructions Ã  la lettre sur machine propre.

---

## Phase 7 â€” Programme R&D : exploiter la redondance, effacer les goulots (NOUVEAU)

> MÃ©ta-principe directeur : **ne jamais payer deux fois pour la mÃªme information**
> (stockage unique + rÃ©fÃ©rences, transfert compressÃ©, copie de ce qui existe dÃ©jÃ ,
> rejeu du calcul dÃ©jÃ  enregistrÃ©).
>
> RÃˆGLES IMPÃ‰RATIVES POUR L'AGENT (lire avant chaque tÃ¢che) :
> 1. Chaque tÃ¢che est une EXPÃ‰RIENCE : hypothÃ¨se â†’ baseline â†’ mesure â†’ verdict.
> 2. TOUJOURS mesurer la baseline AVANT de modifier le code, avec la commande exacte donnÃ©e.
> 3. TOUJOURS vÃ©rifier la non-rÃ©gression de qualitÃ© : gÃ©nÃ©ration greedy (do_sample=False,
>    temperature ignorÃ©e) de 256 tokens sur le prompt fixe
>    "Write a Python function that parses a CSV file and returns a dict."
>    AVANT/APRÃˆS â€” comparer token par token, ET perplexitÃ© sur wikitext-2 (script existant
>    de benchmarks/ ou `evaluate` HF). Ã‰cart de perplexitÃ© > 1% relatif = Ã‰CHEC.
> 4. Chaque expÃ©rience produit un rapport `benchmarks/results/phase7/T7X_report.md` :
>    hypothÃ¨se, commandes exÃ©cutÃ©es, chiffres bruts, verdict (SUCCÃˆS/Ã‰CHEC/AMBIGU).
> 5. Respecter la RÃˆGLE D'ARRÃŠT de chaque tÃ¢che. Un Ã©chec propre et documentÃ© est un
>    rÃ©sultat valide. NE PAS s'acharner au-delÃ  de la rÃ¨gle d'arrÃªt.
> 6. Une branche git par tÃ¢che : `phase7/T7X-nom-court`. Jamais de merge direct sur main
>    sans le rapport.
> 7. Interdiction de modifier les scripts de benchmark existants pour "faire passer"
>    un rÃ©sultat. Si un script doit changer, le documenter dans le rapport.

### T7.0 â€” Prompt/prefix caching pour les agents de codage (gain rapide, risque nul)
**HypothÃ¨se** : les agents de codage renvoient le mÃªme long prÃ©fixe (system prompt + contexte repo) Ã  chaque appel ; rÃ©utiliser le KV cache du prÃ©fixe rÃ©duit massivement le TTFT.
**Ã‰tapes exactes** :
1. Backend llama.cpp : vÃ©rifier que le serveur est lancÃ© avec le cache de prompt actif (llama-cpp-python : paramÃ¨tres `cache_prompt`/slots selon la version installÃ©e â€” lire la doc de la version avec `pip show llama-cpp-python` puis le README correspondant sur GitHub).
2. Backend HF : implÃ©menter la rÃ©utilisation de `past_key_values` quand le nouveau prompt commence par le prÃ©fixe du prÃ©cÃ©dent (comparaison des token ids ; tronquer le cache au point de divergence).
3. Test : envoyer 2 requÃªtes successives partageant un prÃ©fixe de 4000 tokens, ne diffÃ©rant que sur les 50 derniers.
**Mesure** : TTFT (time-to-first-token) requÃªte 2, avec/sans cache.
**Acceptation** : TTFT requÃªte 2 rÃ©duit d'au moins 50% ; sorties greedy identiques avec/sans cache.
**RÃ¨gle d'arrÃªt** : si l'implÃ©mentation HF dÃ©passe 2 jours, ne livrer que la voie llama.cpp.

### T7.1 â€” Prompt lookup decoding (speculation par n-grammes, sans modÃ¨le draft)
**HypothÃ¨se** : en gÃ©nÃ©ration de code, la sortie recopie souvent des n-grammes dÃ©jÃ  prÃ©sents dans le contexte ; les accepter par blocs accÃ©lÃ¨re le dÃ©codage sans modÃ¨le draft.
**Ã‰tapes exactes** :
1. Backend HF : dans le chemin de gÃ©nÃ©ration (`inference_pipeline.py` ou Ã©quivalent), exposer le paramÃ¨tre natif de transformers `prompt_lookup_num_tokens` (supportÃ© par `model.generate`). Le cÃ¢bler Ã  une env var `VRM_PROMPT_LOOKUP=N` (dÃ©faut 0 = dÃ©sactivÃ©, valeur recommandÃ©e 10).
2. VÃ©rifier compatibilitÃ© avec le split multi-GPU (la spÃ©culation utilise le mÃªme modÃ¨le ; aucun second modÃ¨le Ã  charger).
3. Bench : prompt de codage long (inclure un fichier source de ~200 lignes dans le prompt et demander une modification), 256 tokens, greedy, sur Qwen2.5-7B puis 14B 2-GPU.
**Mesure** : tok/s avec VRM_PROMPT_LOOKUP=0 vs =10. Sorties greedy identiques exigÃ©es (la spÃ©culation est sans perte par construction â€” si les sorties diffÃ¨rent, il y a un bug).
**Acceptation** : gain â‰¥ 20% tok/s sur le prompt de codage ; aucun changement de sortie ; rapport avec les deux chiffres.
**RÃ¨gle d'arrÃªt** : si gain < 10% sur 3 prompts de code diffÃ©rents, documenter et fermer (rÃ©sultat nÃ©gatif valide).

### T7.2 â€” Compression FP8 des activations Ã  la frontiÃ¨re inter-GPU
**HypothÃ¨se** : quantifier les activations en FP8 (e4m3) uniquement pour le transfert 3090â†”5070 Ti divise par 2 les octets transfÃ©rÃ©s avec une perte de qualitÃ© < 1% de perplexitÃ©.
**Ã‰tapes exactes** :
1. Localiser le point de transfert des activations dans le code de pipeline multi-GPU (chercher l'appel `.to(device)` ou le TransferManager entre blocs â€” fichiers candidats : `core/transfer_manager.py`, `core/inference_pipeline.py`).
2. ImplÃ©menter `core/boundary_codec.py` avec exactement deux fonctions :
   ```python
   def encode_fp8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
       # scale par tenseur: s = t.abs().amax() / 448.0 (max e4m3); Ã©viter division par zÃ©ro
       # retourne (t_fp8 = (t/s).to(torch.float8_e4m3fn), s)
   def decode_fp8(t_fp8, s, dtype) -> torch.Tensor:
       # retourne (t_fp8.to(dtype) * s)
   ```
3. Activer derriÃ¨re `VRM_BOUNDARY_FP8=1`. L'encode se fait sur le GPU source AVANT transfert, le decode sur le GPU destination APRÃˆS.
4. Si la perplexitÃ© dÃ©passe le seuil avec une scale par tenseur, essayer scale par canal (amax sur dim=-1, garder le vecteur de scales en FP16 â€” surcoÃ»t de transfert nÃ©gligeable).
**Mesure** : (a) octets transfÃ©rÃ©s par token (logger les tailles), (b) tok/s 14B 2-GPU avec/sans, (c) perplexitÃ© wikitext-2 avec/sans, (d) sortie greedy avant/aprÃ¨s (peut diffÃ©rer lÃ©gÃ¨rement â€” c'est la perplexitÃ© qui tranche ici).
**Acceptation** : octets/token rÃ©duits d'environ 50% ; tok/s â‰¥ baseline (le gain dÃ©pend de la part du transfert dans le temps total â€” mÃªme +5% est un succÃ¨s si qualitÃ© OK) ; perplexitÃ© dÃ©gradÃ©e de < 1% relatif.
**RÃ¨gle d'arrÃªt** : si perplexitÃ© > 1% mÃªme en per-channel, tester FP16â†’ rien Ã  gagner ; fermer avec rapport. NE PAS tenter INT4 dans cette tÃ¢che (c'est T7.3).

### T7.3 â€” Rotation d'incohÃ©rence absorbÃ©e Ã  la frontiÃ¨re (dÃ©pend du SUCCÃˆS de T7.2)
**HypothÃ¨se** : appliquer une rotation de Hadamard aux activations Ã  la coupure du pipeline (absorbÃ©e dans les poids adjacents, coÃ»t runtime nul) Ã©tale les outliers et permet de passer le transfert de FP8 Ã  INT4 sans dÃ©passer 1% de perplexitÃ©.
**Contexte mathÃ©matique pour l'agent** : si H est orthogonale (H @ H.T = I), alors pour la couche linÃ©aire de sortie du dernier bloc du GPU 0 de poids W0 et la premiÃ¨re couche du GPU 1 de poids W1 : remplacer W0 par (H @ W0) et W1 par (W1 @ H.T) ne change PAS la fonction calculÃ©e, mais l'activation transmise devient H @ a au lieu de a â€” mieux conditionnÃ©e pour la quantization. Utiliser une matrice de Hadamard normalisÃ©e H/sqrt(n) de taille hidden_size (si hidden_size n'est pas une puissance de 2, utiliser un Hadamard par blocs sur la plus grande puissance de 2 divisant la dimension, identitÃ© sur le reste).
**Ã‰tapes exactes** :
1. Ã‰tendre `core/boundary_codec.py` : `apply_boundary_rotation(model, split_point)` qui modifie les deux matrices de poids in-place au chargement (une seule fois), derriÃ¨re `VRM_BOUNDARY_ROTATE=1`.
2. ATTENTION aux subtilitÃ©s : ne pas casser les biais (b0 devient H @ b0) ; vÃ©rifier qu'aucune normalisation (RMSNorm/LayerNorm) ne se trouve ENTRE la sortie de W0 et le transfert â€” si oui, choisir un autre point d'absorption ou documenter l'impossibilitÃ©.
3. Validation d'identitÃ© D'ABORD : avec rotation activÃ©e et transfert en BF16 (sans quantization), la sortie greedy doit Ãªtre identique Ã  la baseline au token prÃ¨s (tolÃ©rance : divergence aprÃ¨s 200+ tokens acceptable pour cause d'arrondi flottant ; avant 50 tokens = bug).
4. Puis : rotation + INT4 au transfert (quantization symÃ©trique par groupe de 128, scales FP16).
**Mesure** : perplexitÃ© et tok/s pour : baseline BF16 / FP8 seul (T7.2) / rotation+INT4.
**Acceptation** : rotation+INT4 avec perplexitÃ© < 1% de dÃ©gradation relative â†’ octets/token rÃ©duits de 75% vs baseline. Publier le tableau des 3 configurations.
**RÃ¨gle d'arrÃªt** : si l'Ã©tape 3 (identitÃ©) Ã©choue aprÃ¨s 3 jours de debug, fermer la tÃ¢che â€” le point d'absorption est probablement mauvais pour cette architecture. RÃ©sultat nÃ©gatif documentÃ© = OK.

### T7.4 â€” Transport adaptatif pilotÃ© par modÃ¨le de coÃ»t
**HypothÃ¨se** : la configuration optimale de transfert (taille de chunk, nombre de buffers, compresser ou non) dÃ©pend de la paire de GPUs et de la taille du tenseur ; un micro-benchmark au dÃ©marrage + une table de dÃ©cision battent toute constante codÃ©e en dur.
**Ã‰tapes exactes** :
1. CrÃ©er `core/transport_profiler.py` : au premier dÃ©marrage (ou si `VRM_REPROFILE=1`), pour chaque paire de GPUs, mesurer la bande passante effective pour chunk âˆˆ {1, 4, 8, 16, 32, 64} MB Ã— buffers âˆˆ {2, 3} Ã— tailles de tenseur âˆˆ {1, 16, 128} MB. DurÃ©e totale du profilage < 60 s. Sauvegarder dans `~/.vramancer/transport_profile.json`.
2. Le TransferManager lit le profil et choisit la config au plus proche voisin de la taille du tenseur Ã  transfÃ©rer.
3. DÃ©cision de compression (si T7.2 a rÃ©ussi) : compresser si `taille/bp_mesurÃ©e > taille/(2*bp_mesurÃ©e) + coÃ»t_encode_mesurÃ©` â€” c'est-Ã -dire si le temps gagnÃ© sur le fil dÃ©passe le coÃ»t d'encodage, mesurÃ© lui aussi au profilage.
**Mesure** : tok/s 14B 2-GPU : config codÃ©e en dur actuelle vs config auto-sÃ©lectionnÃ©e.
**Acceptation** : l'auto-sÃ©lection fait au moins jeu Ã©gal avec la meilleure config manuelle connue (jamais pire de plus de 2%) et la sÃ©lection est visible dans les logs au niveau INFO.
**RÃ¨gle d'arrÃªt** : aucune (tÃ¢che d'ingÃ©nierie sÃ»re, pas une expÃ©rience risquÃ©e) ; budget 1 semaine.

### T7.5 â€” Test Ã©clair : encodage diffÃ©rentiel temporel des activations (1 journÃ©e MAX)
**HypothÃ¨se (incertaine, Ã  tuer vite si fausse)** : les Ã©tats cachÃ©s Ã  la frontiÃ¨re Ã©voluent peu d'un token de dÃ©codage au suivant ; transmettre le delta quantifiÃ© coÃ»terait moins que l'Ã©tat complet.
**Ã‰tapes exactes** :
1. Script jetable `benchmarks/probe_delta.py` : gÃ©nÃ©rer 100 tokens (greedy, Qwen2.5-7B), logger l'activation Ã  la frontiÃ¨re Ã  chaque step, calculer `r_t = norm(h_t - h_{t-1}) / norm(h_t)` pour t=2..100.
2. Rapporter min/mÃ©diane/max de r_t.
**Verdict** : mÃ©diane < 0.3 â†’ hypothÃ¨se vivante, ouvrir une tÃ¢che de suivi ; mÃ©diane â‰¥ 0.3 â†’ fermer dÃ©finitivement avec le chiffre dans le rapport.
**RÃ¨gle d'arrÃªt** : 1 journÃ©e, c'est un test de faisabilitÃ©, PAS une implÃ©mentation.

### T7.6 â€” Auto-heal du serveur d'infÃ©rence (idÃ©e produit, risque nul, haute valeur)
**HypothÃ¨se** : un serveur qui se rÃ©tablit seul d'un OOM ou d'un crash backend est indispensable pour un usage "agent de codage branchÃ© toute la journÃ©e".
**Ã‰tapes exactes** :
1. Dans le chemin de gÃ©nÃ©ration du serveur (`production_api.py` / pipeline), implÃ©menter l'Ã©chelle de rÃ©cupÃ©ration OOM, dans cet ordre strict, avec log WARN Ã  chaque barreau :
   a. `torch.cuda.empty_cache()` + retry de la requÃªte courante (1 fois) ;
   b. rÃ©duire le contexte max acceptÃ© de 25% pour les requÃªtes suivantes ;
   c. si OOM au CHARGEMENT : re-splitter avec 10% de marge VRAM en plus par GPU ;
   d. dernier recours : recharger en NF4 et exposer l'Ã©tat dÃ©gradÃ© dans `/health` (`"degraded": true, "reason": "oom_fallback_nf4"`).
2. Watchdog : si le process de gÃ©nÃ©ration ne rÃ©pond pas en `VRM_GENERATE_TIMEOUT` (existe dÃ©jÃ  d'aprÃ¨s le CHANGELOG), tuer la gÃ©nÃ©ration en cours, libÃ©rer, rÃ©pondre 503 avec message clair â€” le serveur ne meurt JAMAIS.
3. Test automatisÃ© `tests/test_autoheal.py` : provoquer un OOM artificiel (allouer un tenseur gÃ©ant pendant une gÃ©nÃ©ration) et vÃ©rifier que (a) le process survit, (b) la requÃªte suivante aboutit, (c) `/health` reflÃ¨te l'Ã©tat.
**Acceptation** : le test passe 5 fois de suite ; aucune fuite (VRAM revient au niveau prÃ©-incident Ã  Â±5%, vÃ©rifiÃ© via `torch.cuda.memory_allocated`).

### T7.7 â€” Boucle d'auto-amÃ©lioration gardÃ©e par benchmark (`vramancer autotune`)
**HypothÃ¨se** : une boucle qui mute les paramÃ¨tres de configuration et ne conserve que les variantes prouvÃ©es plus rapides par le banc constitue une auto-amÃ©lioration sÃ»re.
**Ã‰tapes exactes** :
1. CrÃ©er `scripts/autotune.py` : espace de recherche = dict de paramÃ¨tres existants UNIQUEMENT (taille de chunk, VRM_PROMPT_LOOKUP, flags torch.compile, VRM_BOUNDARY_FP8 si T7.2 okâ€¦). AUCUNE mutation de code source â€” seulement des env vars / configs.
2. Boucle : config champion = config actuelle â†’ gÃ©nÃ©rer 1 mutation alÃ©atoire d'1 paramÃ¨tre â†’ lancer `benchmarks/bench_tok_s.py` (3 rÃ©pÃ©titions, garder la mÃ©diane) â†’ si mÃ©diane > champion de plus de 2% (seuil anti-bruit), la mutation devient champion â†’ logger dans `benchmarks/results/phase7/autotune_history.jsonl`.
3. Budget : `--iterations N` (dÃ©faut 20). Ã€ la fin, Ã©crire la config champion dans `~/.vramancer/tuned.env` + un rapport.
**Acceptation** : la boucle tourne 20 itÃ©rations sans crash ; le champion final est â‰¥ champion initial (la boucle ne peut PAS rÃ©gresser par construction â€” vÃ©rifier cette propriÃ©tÃ© dans un test).
**Garde-fou ABSOLU** : l'autotune ne touche jamais au code, jamais aux poids du modÃ¨le, jamais aux scripts de bench.

### T7.8 â€” [EXPÃ‰RIMENTAL â€” NE PAS LANCER SANS ACCORD EXPLICITE DE L'UTILISATEUR] LoRA nocturne sur l'usage
**HypothÃ¨se** : fine-tuner pÃ©riodiquement un petit LoRA sur les sorties de codage explicitement acceptÃ©es par l'utilisateur amÃ©liore le modÃ¨le local sur SON style/SES repos.
**Garde-fous obligatoires AVANT toute implÃ©mentation** : (a) collecte opt-in uniquement (`VRM_COLLECT_ACCEPTED=1`), donnÃ©es stockÃ©es localement en clair listables/effaÃ§ables par l'utilisateur ; (b) barriÃ¨re d'Ã©valuation : le LoRA candidat doit faire â‰¥ baseline sur un mini-jeu d'Ã©val fixe (20 problÃ¨mes de code dÃ©finis une fois, jamais modifiÃ©s) ET perplexitÃ© gÃ©nÃ©rale dÃ©gradÃ©e < 2% ; (c) bascule atomique avec rollback en une commande ; (d) jamais de modification des poids de base â€” LoRA sÃ©parÃ©, dÃ©sactivable par env var.
**RÃ¨gle d'arrÃªt** : si la barriÃ¨re d'Ã©valuation Ã©choue 3 cycles de suite, suspendre la fonctionnalitÃ©.

### T7.9 â€” Entrelacement de deux requÃªtes dans le pipeline (combler la bulle)
**HypothÃ¨se** : en pipeline 2-GPU avec une requÃªte unique, chaque GPU est oisif pendant que l'autre calcule (~50% d'oisivetÃ©) ; traiter deux requÃªtes dÃ©calÃ©es d'un Ã©tage double presque le dÃ©bit agrÃ©gÃ© sans modifier le modÃ¨le.
**Ã‰tapes exactes** :
1. PrÃ©-mesure de la bulle : instrumenter le pipeline (timestamps dÃ©but/fin de calcul par GPU sur 50 tokens, 1 requÃªte) et calculer le taux d'oisivetÃ© par GPU. Mettre les chiffres dans le rapport â€” si l'oisivetÃ© est < 30%, la tÃ¢che rapportera peu : le noter et demander confirmation avant de continuer.
2. ImplÃ©menter dans le serveur (pas dans le CLI) un ordonnanceur Ã  2 slots : quand 2 requÃªtes sont en file, la requÃªte B exÃ©cute ses blocs GPU 0 pendant que la A exÃ©cute ses blocs GPU 1, en alternance stricte. Chaque requÃªte garde son propre KV cache. Synchronisation par events CUDA, PAS par locks Python autour des forward (le GIL est relÃ¢chÃ© pendant les kernels).
3. Limiter Ã  2 requÃªtes simultanÃ©es (`VRM_PIPELINE_SLOTS=2`, dÃ©faut 1 = comportement actuel).
**Mesure** : dÃ©bit agrÃ©gÃ© (tokens totaux/seconde) avec 2 requÃªtes concurrentes : slots=1 (sÃ©quentiel) vs slots=2 (entrelacÃ©), sur 14B 2-GPU. Latence par requÃªte aussi (elle ne doit pas exploser : < +30% vs requÃªte seule).
**Acceptation** : dÃ©bit agrÃ©gÃ© â‰¥ +60% avec slots=2 ; sorties greedy de chaque requÃªte identiques au mode sÃ©quentiel ; aucune corruption croisÃ©e des KV caches (test : 2 prompts trÃ¨s diffÃ©rents, vÃ©rifier les 2 sorties).
**RÃ¨gle d'arrÃªt** : si aprÃ¨s 1 semaine les sorties sont corrompues par interfÃ©rence des caches/streams, fermer et documenter â€” c'est un signal que l'architecture actuelle du pipeline ne s'y prÃªte pas sans refonte.

### T7.10 â€” Capture CUDA Graph du pas de dÃ©codage
**HypothÃ¨se** : chaque token de dÃ©codage relance la mÃªme sÃ©quence de kernels ; capturer le graphe une fois et le rejouer Ã©limine l'overhead de lancement Python/CUDA (gain surtout visible sur les petits/moyens modÃ¨les et en multi-GPU oÃ¹ les lancements s'accumulent).
**Ã‰tapes exactes** :
1. Voie simple d'abord : vÃ©rifier si `torch.compile(model, mode="reduce-overhead")` (qui utilise les CUDA Graphs en interne) est dÃ©jÃ  actif dans TurboEngine. Si oui, mesurer ce qu'il capture rÃ©ellement en multi-GPU (souvent : il refuse de capturer Ã  travers les transferts inter-device â€” le vÃ©rifier avec `TORCH_LOGS=graph_breaks`).
2. Si la capture casse Ã  la frontiÃ¨re inter-GPU : capturer DEUX graphes sÃ©parÃ©s (blocs GPU 0, blocs GPU 1) avec buffers d'entrÃ©e/sortie statiques prÃ©-allouÃ©s, le transfert restant hors graphe. Les KV caches doivent Ãªtre Ã  adresses fixes (prÃ©-allouer Ã  la taille max du contexte â€” vÃ©rifier la compatibilitÃ© avec le paged attention existant ; si incompatible, documenter le conflit et s'arrÃªter).
3. Activer derriÃ¨re `VRM_CUDA_GRAPHS=1`.
**Mesure** : tok/s sur TinyLlama-1.1B (oÃ¹ l'overhead de lancement pÃ¨se le plus), Qwen2.5-7B, et 14B 2-GPU, avec/sans.
**Acceptation** : gain â‰¥ 10% sur au moins un des trois modÃ¨les, sorties greedy identiques, pas de rÃ©gression sur les autres.
**RÃ¨gle d'arrÃªt** : les CUDA Graphs sont notoirement fragiles (adresses figÃ©es, pas d'allocation pendant la capture). Budget 1 semaine ; au premier conflit insoluble avec le paged attention, fermer proprement avec le diagnostic.

### T7.11 â€” Cache d'experts chauds pour MoE (la version concrÃ¨te de l'idÃ©e "cerveau")
**HypothÃ¨se** : sur un MoE comme Qwen3.6-35B-A3B, la distribution d'activation des experts est trÃ¨s inÃ©gale et localisÃ©e ; garder les experts frÃ©quents ("chauds") en VRAM et les rares en RAM CPU avec chargement Ã  la demande permet de servir un MoE plus gros que la VRAM disponible, avec une pÃ©nalitÃ© de latence bornÃ©e.
**Ã‰tapes exactes** :
1. Ã‰TAPE DE MESURE D'ABORD (1-2 jours, ne rien implÃ©menter avant) : script `benchmarks/probe_expert_usage.py` â€” charger le modÃ¨le MoE, hooker le routeur, gÃ©nÃ©rer 2000 tokens sur 5 prompts de codage variÃ©s, logger la frÃ©quence d'activation par expert et par couche. Rapporter : % des activations couvert par le top 25% / top 50% des experts.
2. VERDICT intermÃ©diaire : si le top 50% des experts couvre < 80% des activations, la localitÃ© est insuffisante â†’ fermer la tÃ¢che avec les chiffres (rÃ©sultat nÃ©gatif valide).
3. Si localitÃ© confirmÃ©e : implÃ©menter `core/expert_cache.py` â€” poids des experts "froids" en RAM CPU (pinned memory), copie vers un buffer VRAM rÃ©servÃ© Ã  la demande, politique LRU, taille du pool configurable `VRM_EXPERT_POOL_GB`. PrÃ©chargement spÃ©culatif optionnel en phase 2 (PAS dans la premiÃ¨re version).
4. IntÃ©gration : backend HF uniquement dans un premier temps. llama.cpp gÃ¨re dÃ©jÃ  son propre offload MoE (option de type `--n-cpu-moe` selon la version) â€” le documenter comme alternative et le benchmarker en COMPARAISON.
**Mesure** : tok/s et VRAM utilisÃ©e : (a) modÃ¨le entier si Ã§a tient, (b) llama.cpp avec offload experts natif, (c) notre expert_cache. MÃªme prompt de codage, greedy.
**Acceptation** : notre cache permet de charger une config qui ne tenait pas en VRAM seule, avec tok/s â‰¥ 70% de la solution llama.cpp Ã©quivalente (si on fait moins bien que llama.cpp, le rapport doit le dire honnÃªtement et recommander llama.cpp pour ce cas).
**RÃ¨gle d'arrÃªt** : tÃ¢che la plus lourde de la phase. Jalon obligatoire Ã  2 semaines : si l'Ã©tape 3 n'est pas fonctionnelle, geler et livrer au minimum l'Ã©tude de localitÃ© (Ã©tape 1) + le comparatif llama.cpp â€” dÃ©jÃ  utile pour la doc.

### T7.12 â€” DÃ©sagrÃ©gation poids/KV : la VRAM du GPU secondaire comme magasin de KV cache
> IdÃ©e : modÃ¨le ENTIER sur le GPU rapide (toute sa VRAM pour les poids), KV cache
> dÃ©bordant sur la VRAM du GPU secondaire via P2P. Supprime la bulle de pipeline
> pour les modÃ¨les qui tiennent sur un seul GPU, et dÃ©bloque les contextes longs.
> PRÃ‰REQUIS : Phase 5 validÃ©e (P2P fonctionnel et intÃ¨gre). Sans P2P, NE PAS FAIRE
> (le double saut via CPU rend le GPU pair pire que la RAM Ã©pinglÃ©e).

**AVERTISSEMENT ARITHMÃ‰TIQUE pour l'agent â€” lire avant d'Ã©crire du code** :
L'attention relit le KV Ã  chaque token. Rapatrier tout le KV distant Ã  chaque Ã©tape
via PCIe (~20 Go/s) contre une lecture locale (~1000 Go/s) = facteur ~50 de pÃ©nalitÃ©
= Ã©chec garanti. La version "stocker ailleurs et tout rapatrier" est INTERDITE.
Seules les versions ci-dessous sont autorisÃ©es.

**Palier 0 â€” Ã‰tude de viabilitÃ© (2 jours MAX, AVANT toute implÃ©mentation)** :
1. Script `benchmarks/probe_kv_disagg.py` : pour Qwen2.5-7B Ã  ctx 8K/32K/64K, calculer
   (a) taille du KV par token et totale, (b) octets que l'attention lit par token,
   (c) temps thÃ©orique de lecture via P2P Ã  la bande passante MESURÃ‰E en Phase 5,
   (d) ratio fenÃªtre chaude locale nÃ©cessaire pour que le trafic distant < 10% du temps par token,
   en supposant que seuls X% des tokens anciens sont consultÃ©s (mesurer X avec un hook
   d'attention : part de la masse d'attention portÃ©e par les 2048 derniers tokens vs le reste).
2. Rapport avec le tableau. Si la masse d'attention sur les tokens anciens est trop
   uniformÃ©ment rÃ©partie (pas de localitÃ©), documenter et passer directement au Palier 2
   (attention prÃ¨s des donnÃ©es) ou fermer.

**Palier 1 â€” Spill des pages froides (si le Palier 0 montre de la localitÃ©)** :
1. Ã‰tendre le paged attention existant : pool de pages local (3090) + pool distant (5070 Ti)
   allouÃ© via `torch.empty(..., device="cuda:1")`, transferts par `copy_` P2P asynchrones
   sur stream dÃ©diÃ©.
2. Politique : fenÃªtre chaude = `VRM_KV_HOT_WINDOW` derniers tokens (dÃ©faut 4096) toujours
   locale ; pages plus anciennes migrÃ©es vers le pair en tÃ¢che de fond ; rapatriement
   par prÃ©chargement (pendant le calcul du token courant, prÃ©charger les pages probables
   du suivant).
3. Mesure : tok/s Ã  ctx 32K et 64K : (a) tout-local si Ã§a tient, (b) spill vers RAM Ã©pinglÃ©e
   (baseline existante), (c) spill vers GPU pair. PerplexitÃ© inchangÃ©e exigÃ©e (le spill ne
   change pas les maths, seulement l'emplacement â€” toute dÃ©rive = bug).
**Acceptation Palier 1** : (c) â‰¥ (b) en tok/s, et permet un contexte qui ne tenait pas en (a).

**Palier 2 â€” Attention prÃ¨s des donnÃ©es (ambitieux, SEULEMENT si Palier 1 validÃ©)** :
1. Principe mathÃ©matique : l'attention se dÃ©compose par segments de KV. Le GPU 1 calcule
   l'attention de la requÃªte courante sur SES pages (sortie partielle + max et somme des
   exponentielles), le GPU 0 sur les siennes, puis fusion par online softmax :
   exactement la mÃªme rÃ©currence (m, l, acc) que le kernel CUDA paged attention existant
   de VRAMancer, appliquÃ©e entre deux rÃ©sultats partiels. Ce qui transite : O(hidden_size)
   par token, PAS le KV.
2. ImplÃ©mentation minimale : rÃ©pliquer les poids de projection Q/K/V/O nÃ©cessaires sur le
   GPU 1 (quelques centaines de Mo), exÃ©cuter l'attention partielle distante en parallÃ¨le
   de la locale (streams), fusionner sur GPU 0.
3. Mesure : tok/s Ã  64K et 128K vs Palier 1. Sortie greedy identique au tout-local exigÃ©e
   (tolÃ©rance arrondi flottant : divergence avant 100 tokens = bug).
**Acceptation Palier 2** : tok/s Ã  128K â‰¥ 80% du tok/s Ã  8K tout-local (l'objectif est que
le contexte long devienne presque gratuit).
**RÃ¨gle d'arrÃªt** : Palier 2 = 3 semaines MAX. C'est de la recherche appliquÃ©e ; un Ã©chec
documentÃ© avec les chiffres du Palier 1 livrÃ© reste un excellent rÃ©sultat.

**Synergie** : le pool de pages distant du Palier 1 est le mÃªme mÃ©canisme que le pool
d'experts de T7.11 â€” factoriser dans un `core/peer_memory_pool.py` commun si les deux
tÃ¢ches sont menÃ©es.


```
T7.6 (auto-heal)      â†’ indÃ©pendant, FAIRE EN PREMIER (valeur produit immÃ©diate)
T7.0 (prefix cache)   â†’ indÃ©pendant, deuxiÃ¨me (sert la Phase 6)
T7.1 (prompt lookup)  â†’ indÃ©pendant, troisiÃ¨me (gain rapide prouvable)
T7.5 (test delta)     â†’ indÃ©pendant, 1 journÃ©e, n'importe quand
T7.9 (entrelacement)  â†’ quatriÃ¨me (gros gain de dÃ©bit serveur, sert la Phase 6)
T7.2 (FP8 boundary)   â†’ cinquiÃ¨me
T7.3 (rotation)       â†’ SEULEMENT si T7.2 = SUCCÃˆS
T7.4 (transport adaptatif) â†’ aprÃ¨s T7.2 (intÃ¨gre la dÃ©cision de compression)
T7.10 (CUDA Graphs)   â†’ aprÃ¨s T7.9 (les deux touchent le pas de dÃ©codage â€” ne PAS mener en parallÃ¨le)
T7.11 (cache experts MoE) â†’ aprÃ¨s la Phase 6 (nÃ©cessite le modÃ¨le MoE installÃ©) ; commencer par l'Ã©tape de mesure
T7.12 (dÃ©sagrÃ©gation poids/KV) â†’ aprÃ¨s Phase 5 (P2P validÃ© OBLIGATOIRE) ; Palier 0 d'abord ; synergie avec T7.11
T7.7 (autotune)       â†’ en dernier (exploite les paramÃ¨tres crÃ©Ã©s par les autres)
T7.8 (LoRA usage)     â†’ JAMAIS sans accord explicite prÃ©alable de l'utilisateur
```

---

## Principes pour les agents de codage

1. Aucune suppression de code : tout module retirÃ© du core va dans `experimental/` ou un repo d'archive.
2. Chaque dÃ©placement de fichier doit Ãªtre suivi d'un run complet de la suite stub (`VRM_MINIMAL_TEST=1 pytest tests/ -q`) â€” zÃ©ro rÃ©gression tolÃ©rÃ©e.
3. Aucun chiffre de performance ne doit Ãªtre Ã©crit dans un document sans rÃ©fÃ©rence Ã  la section correspondante de `benchmarks/BENCHMARK_RESULTS.md`.
4. Les messages de commit suivent le format `phase/tÃ¢che: description` (ex. `P2/T2.1: move aitp_protocol to experimental/`).
5. Une PR par tÃ¢che, pas de PR fourre-tout.
