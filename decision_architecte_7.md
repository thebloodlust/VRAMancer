# DÃ©cision de l'architecte â€” RÃ©ponse au compte-rendu Phase 7 (Opus/DeepSeek)

> De : l'architecte Â« Fable Â» Â· Ã€ : Claude Opus (exÃ©cution) et DeepSeek (audit)
> En rÃ©ponse Ã  : `COMPTE_RENDU_ARCHITECTE_PHASE7_DEEPSEEK.md` (Ã©tat 2026-06-12)
> Ce document tranche la dÃ©cision A/B/C demandÃ©e, donne les instructions de merge,
> acte les consÃ©quences des mesures de la Partie D, et liste les questions pour DeepSeek.

---

## 1. Validation des rÃ©sultats Phase 7

| TÃ¢che | DÃ©cision de l'architecte |
|---|---|
| T7.1 prompt lookup (+323/+501/+563% lossless) | **VALIDÃ‰ â€” merger sur main.** RÃ©sultat trÃ¨s au-delÃ  du critÃ¨re (+20%). AprÃ¨s merge : promouvoir dans le README comme feature prouvÃ©e, avec lien vers T7.1_report.md. |
| T7.6 auto-heal | **VALIDÃ‰ â€” merger sur main.** |
| T7.0 prefix cache (Ã©chec root-causÃ©) | **ACCEPTÃ‰.** Merger le rapport seul. Note : pour la Phase 6, le caching de prÃ©fixe reste Ã  couvrir via le backend llama.cpp (cache natif), pas via HF. |
| T7.5 delta temporel (mÃ©diane 0.879, fermÃ©) | **ACCEPTÃ‰ â€” fermeture dÃ©finitive.** Le test Ã©clair a fonctionnÃ© exactement comme conÃ§u : hypothÃ¨se tuÃ©e en une journÃ©e avec un chiffre. Merger le rapport. |
| T7.9 Ã©tape 1 (oisivetÃ© 68.7%/47.8%) | **VALIDÃ‰.** La bulle est confirmÃ©e comme le plus gros gisement de perf du projet. L'Ã©tape 2 dÃ©pend de la dÃ©cision Â§3 ci-dessous. |
| Bug collatÃ©ral : Ã©quilibrage device_map NF4 14B (OOM GPU0) | **Ouvrir une issue dÃ©diÃ©e** avec repro. Ne pas corriger en douce dans une autre branche. |

## 2. ConsÃ©quences de la Partie D (mesure PCIe rÃ©elle)

1. **Phase 5 close avec verdict** : P2P = NS entre RTX 3090 et RTX 5070 Ti (consumer).
   Le "20 GB/s" Ã©voquÃ© Ã©tait le transport CPU-staged pipelinÃ©, dÃ©sormais mesurÃ©
   proprement Ã  **25.3 GB/s (+143% vs torch naÃ¯f, 78% du plafond PCIe)**.
   Ã€ documenter dans BENCHMARK_RESULTS.md : section "Transfer strategies â€” measured",
   avec la conclusion honnÃªte "P2P unsupported on this consumer pair; pipelined
   staged transport reaches 78% of PCIe ceiling".
2. **T7.12 (KV cache sur VRAM du pair) â†’ CONGELÃ‰.** Son prÃ©requis explicite (P2P
   fonctionnel) n'est pas rempli ; sans P2P, la VRAM distante est dominÃ©e par la RAM
   Ã©pinglÃ©e. RÃ©activable si un futur matÃ©riel/driver rÃ©tablit le P2P.
3. **chunk_mb=4** (optimal mesurÃ©) = premiÃ¨re entrÃ©e du profil de transport T7.4.
   Conserver la donnÃ©e brute pour l'auto-tuner.
4. Corriger le nommage `bandwidth_gbps` (bits) vs `bandwidth_gbs` (octets) dans
   l'API Rust â€” piÃ¨ge de mesure documentÃ©, fix trivial, Ã  inclure dans le prochain
   commit hardening.

## 3. DÃ‰CISION : cluster frontiÃ¨re (T7.2 / T7.3 / T7.4 / T7.9-Ã©tape2)

**Option retenue : A, en deux paliers, avec porte de sortie vers B.**

Justification (deux chiffres du compte-rendu lui-mÃªme) :
- La bulle mesurÃ©e (68.7%/47.8% d'oisivetÃ©) est le plus gros gain potentiel du
  projet et n'est exploitable qu'en contrÃ´lant le pipeline (Path 2).
- Le GpuPipeline transfÃ¨re Ã  25.3 GB/s lÃ  oÃ¹ le `.to()` naÃ¯f d'accelerate fait
  10.4 GB/s : Path 2 part avec +143% d'avantage de transfert.
- StratÃ©giquement : choisir B ou C acterait que VRAMancer reste dÃ©finitivement un
  wrapper accelerate/llama.cpp. Toute la roadmap diffÃ©renciante (FP8 frontiÃ¨re,
  rotation, transport adaptatif, entrelacement) exige de possÃ©der le pipeline.

### Palier A1 â€” Porte de paritÃ© (budget : 2 semaines MAX)
- Benchmark `Path 2 + GpuPipeline` vs `Path 1 accelerate` sur Qwen2.5-14B bf16 2-GPU.
- CritÃ¨res de passage : (a) sorties greedy identiques token pour token (256 tokens,
  prompt fixe du protocole Phase 7) ; (b) tok/s Path 2 â‰¥ tok/s accelerate âˆ’ 5%.
- Livrable : `benchmarks/results/phase7/A1_path2_vs_accelerate.md` avec les deux
  chiffres et le diff de sorties.

### Palier A2 â€” Bascule (uniquement si A1 passe)
- Path 2 devient le chemin bf16 multi-GPU derriÃ¨re `VRM_PIPELINE=native`
  (opt-in d'abord ; dÃ©faut aprÃ¨s une semaine sans incident, auto-heal T7.6 actif).
- DÃ©blocage dans cet ordre : **T7.9-Ã©tape2** (le plus gros gain attendu) â†’
  **T7.2** (FP8 frontiÃ¨re) â†’ **T7.4** (transport adaptatif, seedÃ© avec chunk=4MB)
  â†’ T7.3 seulement si T7.2 = SUCCÃˆS.

### Porte de sortie
- Si A1 Ã©choue et n'est pas rÃ©parable dans le budget : repli sur **B ciblÃ©**
  (test FP8 par hook sur le module-frontiÃ¨re, faible coÃ»t, mesure seule) et
  **fermeture documentÃ©e** de T7.9-Ã©tape2. Pas d'acharnement.

## 4. Instructions de merge sur main

Merger, aprÃ¨s passage complet de la suite stub (`VRM_MINIMAL_TEST=1 pytest tests/ -q`) :
1. `phase7/T7.1-prompt-lookup`
2. `phase7/T7.6-autoheal`
3. Branches de rapports : T7.0, T7.5, note frontiÃ¨re (T7.2)
4. `hardening/rust-p0-network` â€” les 8 items P0/P2 sont approuvÃ©s ; les refus de
   la section B.4 sont justifiÃ©s (imports prod vÃ©rifiÃ©s) ; les corrections faites
   aux dÃ©tails DeepSeek (features tokio, panic="abort" incompatible pyo3) sont
   techniquement correctes.

Ne PAS merger : tout code du cluster frontiÃ¨re avant le verdict A1.

## 5. HygiÃ¨ne (rappel â€” toujours en attente)

- RÃ©soudre la contradiction AITP : si gelÃ©, `tests/test_aitp.py` part en
  `experimental/` avec les modules (ou marqueur pytest `experimental`).
- DÃ©placer ce compte-rendu et le prÃ©sent document dans `docs/sessions/`,
  supprimer `Fable.md` (plan pÃ©rimÃ©), renommer `7.md` â†’ `docs/PLAN.md`.
- La Phase 0 du plan (LICENSE, racine, requirements) n'est toujours pas faite :
  ce sont les tÃ¢ches les moins chÃ¨res et les plus visibles du repo. Ã€ caser.

## 6. Questions pour DeepSeek (revue CUDA / Rust / Tokio)

Q1. **Pointeurs CUDA Ã  travers les `.await`** : dans `GpuPipeline` et `GpuNetBridge`,
des `CUdeviceptr` ou buffers pinned sont-ils dÃ©tenus Ã  travers des points `.await` ?
Si une tÃ¢che Tokio est annulÃ©e (timeout P0.2) entre l'allocation et la libÃ©ration,
qui libÃ¨re ? Auditer chaque chemin d'annulation pour fuite VRAM/pinned.

Q2. **Design RAII (P2.1)** : proposer les types wrappers (DevicePtr, PinnedBuf,
StreamGuard) avec leurs impls Drop, et statuer sur Send/Sync â€” un CUstream est-il
jamais partagÃ© entre threads Tokio sans synchronisation ? Justifier chaque
`unsafe impl Send/Sync` s'il y en a.

Q3. **shared_runtime (P0.3)** : risque de `block_on` imbriquÃ© (panic "cannot start
a runtime from within a runtime") si un appel pyo3 arrive depuis un thread dÃ©jÃ 
dans le runtime ? Et le GIL : les chemins de transfert relÃ¢chent-ils bien le GIL
(`py.allow_threads`) pendant les copies, sinon le serveur Python gÃ¨le pendant
chaque transfert de 256 MB.

Q4. **Cycle de vie des buffers pinned du triple-buffering** : sont-ils allous une
fois et rÃ©utilisÃ©s, ou re-pinnÃ©s Ã  chaque appel ? (cudaHostAlloc/Register est
coÃ»teux ; le re-pinning par appel ruinerait les 25 GB/s sur petits tenseurs.)
Mesurer le coÃ»t du premier appel vs rÃ©gime Ã©tabli.

Q5. **Annulation et empoisonnement** : si un transfert Ã©choue Ã  mi-course (timeout
I/O), l'Ã©tat du pipeline (buffers en vol, events CUDA non signalÃ©s) est-il
rÃ©cupÃ©rable ou faut-il recrÃ©er le GpuPipeline ? Proposer un test de chaos.

Q6. **VÃ©rifier la claim AVX-512 (B.4/P2.5)** : confirmer avec `cargo asm` (ou
objdump) sur le CPU cible que la boucle XOR auto-vectorise rÃ©ellement en AVX-512 ;
si le CPU de prod ne supporte qu'AVX2, la claim doit Ãªtre reformulÃ©e.

Q7. **Instrumentation pour A1** : exposer cÃ´tÃ© Rust des compteurs (octets
transfÃ©rÃ©s, temps DtoH/HtoD/overlap effectif) lisibles depuis Python, pour que le
benchmark A1 dÃ©compose le temps par token entre calcul et transfert.

Q8. **unwrap()/panic Ã  la frontiÃ¨re FFI** : passe en revue des `unwrap`/`expect`
atteignables depuis pyo3 ; chaque panic doit devenir une exception Python propre,
jamais un abort (cohÃ©rent avec le refus de panic="abort").

---

*Une fois A1 livrÃ© et les merges faits, mettre Ã  jour le compte-rendu consolidÃ©
et le figer dans `docs/sessions/`. â€” L'architecte*
