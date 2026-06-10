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

## Principes pour les agents de codage

1. Aucune suppression de code : tout module retirÃ© du core va dans `experimental/` ou un repo d'archive.
2. Chaque dÃ©placement de fichier doit Ãªtre suivi d'un run complet de la suite stub (`VRM_MINIMAL_TEST=1 pytest tests/ -q`) â€” zÃ©ro rÃ©gression tolÃ©rÃ©e.
3. Aucun chiffre de performance ne doit Ãªtre Ã©crit dans un document sans rÃ©fÃ©rence Ã  la section correspondante de `benchmarks/BENCHMARK_RESULTS.md`.
4. Les messages de commit suivent le format `phase/tÃ¢che: description` (ex. `P2/T2.1: move aitp_protocol to experimental/`).
5. Une PR par tÃ¢che, pas de PR fourre-tout.
