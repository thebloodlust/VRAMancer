# Compte-rendu consolidé — Phase 7 (architecte « fable ») + revue croisée DeepSeek

> Destiné à **l'architecte** (programme R&D Phase 7) **et à DeepSeek** (réponse à
> ses audits). État au **2026-06-12**. Document vivant — sera mis à jour à mesure
> que Phase 7 avance.
>
> Auteur : Claude Opus. Méthode : chaque affirmation vérifiée contre le code réel
> de l'arbre actif (`/home/jeremie/VRAMancer/VRAMancer/`).

---

> ## 📍 Placement de ce document
>
> Ce rapport est posé directement à la racine de `main`, à côté de `Fable.md` et
> `7.md`, pour être visible immédiatement par l'architecte « fable » et par
> DeepSeek. **Ce commit n'ajoute que ce fichier — aucun code n'est mergé sur
> `main`.** Le code réel reste sur ses branches dédiées, conformément à la règle 6
> de `7.md` (« une branche par tâche, jamais de merge direct sur `main` sans le
> rapport ») : ce document est précisément ce rapport, condition préalable à toute
> décision de merge.
>
> **Où trouver le code correspondant** :
> - `hardening/rust-p0-network` — 3 commits de correctifs P0 réseau Rust (bornes
>   anti-OOM, timeouts Tokio, runtime partagé, binaire -42%), + `SUPERAUDIT.md` et
>   `reponse_a_opus.md` (audits DeepSeek), + ce rapport et le bench P2.10 (Partie D).
> - `phase7/T7.0-prefix-cache`, `phase7/T7.1-prompt-lookup`,
>   `phase7/T7.2-boundary-fp8`, `phase7/T7.5-delta-probe`, `phase7/T7.6-autoheal`,
>   `phase7/T7.9-pipeline-interleave` — une branche par tâche Phase 7, chacune avec
>   son rapport `T7X_report.md` dans `benchmarks/results/phase7/`.
>
> **Ce qui attend l'architecte** : la décision A/B/C sur le cluster « frontière »
> (Partie A ci-dessous), puis indiquer quelles branches merger sur `main`.

---

## Synthèse exécutive

- **Phase 7** : 5 tâches traitées (T7.6 ✅, T7.0 ❌, T7.1 ✅ +500 % tok/s lossless,
  T7.5 ⛔ négatif, T7.9 étape 1 ✅). **1 décision t'attend** : A/B/C sur le cluster
  frontière (T7.2/3/4/9) qui cible un chemin de code **inactif en prod** (Partie A).
- **DeepSeek** : son 1er audit visait un checkout périmé (erreurs corrigées) ; sa
  ré-analyse `SUPERAUDIT.md` est juste. J'ai implémenté **8 items** sûrs (P0 sécurité
  + build, binaire -42 %), corrigé 3 de ses détails erronés, et **refusé** ce qui
  casserait la prod/les tests (preuves, Partie B).
- **1ère mesure sur matériel réel** (Partie D) : transfert 3090↔5070 Ti — le pipeline
  VRAMancer atteint **25 GB/s (+143 % vs torch naïf)**, optimum `chunk=4 MB`. Code
  validé, chiffres physiquement cohérents.
- **Restant** : ta décision A/B/C, puis T7.11/T7.10 ; multi-nœud testable au coup
  par coup (laptop/Mac branchés à la demande) ; cross-vendor bus → en attente d'un GPU AMD.

---

# PARTIE A — Phase 7 (pour l'architecte)

Règles suivies : chaque tâche = expérience (hypothèse → baseline → mesure →
verdict) ; rapport par tâche dans `benchmarks/results/phase7/` ; une branche par
tâche ; un échec/fermeture documenté est un résultat valide ; jamais modifier un
script de bench pour « faire passer » un résultat.

## Tâches traitées

| Tâche | Verdict | Résumé | Rapport |
|---|---|---|---|
| **T7.6** auto-heal serveur | ✅ **SUCCÈS** | Échelle de récupération OOM + watchdog ; le serveur ne meurt jamais. Validé. | `T7.6_*` |
| **T7.0** prefix cache | ❌ **ÉCHEC** (root-causé) | Gain absent / négatif, cause identifiée et documentée. | `T7.0_*` |
| **T7.1** prompt lookup (`VRM_PROMPT_LOOKUP`) | ✅ **SUCCÈS** | 7B greedy : **+323 % / +501 % / +563 % tok/s**, **lossless** (256 tokens identiques au token près). Découverte : 14B/2-GPU bloqué par un bug préexistant d'équilibrage `device_map` sous NF4 (OOM GPU0, GPU1 sous-utilisé) — **pas** la faute de T7.1. | `T7.1_report.md`, `T7.1_14b_attempt_oom.*` |
| **T7.5** delta temporel frontière (test éclair) | ⛔ **FERMÉ** (négatif) | Médiane `r_t = 0.879` (≫ seuil 0.3) : l'activation frontière change de ~88 % de sa norme par token. Encodage différentiel temporel non viable. Aucune suite. | `T7.5_report.md` |
| **T7.9** entrelacement pipeline — **étape 1** (pré-mesure bulle) | ✅ **BULLE_SIGNIFICATIVE** | Oisiveté GPU **68.7 % / 47.8 %** (seuil 30 %). Gain agrégat théorique ≤ ~1.9×. **Étape 2 (scheduler 2-slots) NON faite** — voir blocage ci-dessous. | `T7.9_report.md` |

## ⚠️ Décision requise de l'architecte — le cluster « frontière » (T7.2/T7.3/T7.4/T7.9-étape2)

**Constat factuel, prouvé par le code** : ces quatre tâches agissent toutes sur le
transfert d'activations inter-GPU `hidden_states.to(block_dev)`
([core/backends.py:1736](core/backends.py)), qui n'est exécuté que dans le
**« Path 2 »** (découpage manuel VRAMancer, `device_map=None`). Or **la production
n'emprunte jamais ce chemin** : le bf16 multi-GPU passe par `device_map="auto"`
(Path 1 / accelerate, `blocks=None`), le NF4 est forcé mono-GPU, et le serveur
courant tourne en llama.cpp. → Détail complet :
`benchmarks/results/phase7/NOTE_ARCHITECTE_boundary_path.md`.

**Trois options (à trancher avant de coder T7.2/3/4 + T7.9-étape2)** :
- **A.** Faire de Path 2 le chemin de prod multi-GPU bf16 (d'abord prouver la
  parité qualité/débit avec accelerate).
- **B.** Garder accelerate, insérer l'encode/decode FP8 (T7.2) via un hook sur le
  module-frontière (technique du probe T7.5). *Reco : B ciblé sur T7.2 comme test
  à faible coût.* T7.9-étape2 (entrelacement) deviendrait infaisable sans contrôle
  explicite des forwards → à fermer.
- **C.** Sortir le cluster frontière de la Phase 7 (résultats négatifs documentés).

## Tâches Phase 7 restantes (indépendantes de la frontière)
- **T7.11** — cache d'experts MoE, **mesure d'abord** (probe fréquence d'activation
  des experts sur Qwen3.6-35B-A3B). Pertinent, peu risqué. *Prochaine étape proposée.*
- **T7.10** — capture CUDA Graph du pas de décodage.
- **T7.7** (autotune) — en dernier. **T7.8** (LoRA) — jamais sans accord explicite.

---

# PARTIE B — Revue croisée des audits DeepSeek

DeepSeek a produit trois documents (dans `/home/jeremie/VRAMancer/`) :
`conseildeepseek.md` (55 reco) + `avisdeepseek.md` (briefing), puis `SUPERAUDIT.md`.

## B.1 — Les deux premiers audits visaient le mauvais arbre

`conseildeepseek.md` / `avisdeepseek.md` ont analysé le **checkout parent**
(`/home/jeremie/VRAMancer/`, périmé : `lib.rs` 497 lignes, `core/` 46 fichiers,
daté mars), pas l'arbre actif (`lib.rs` ~2150 lignes, `core/` 73 fichiers +
`experimental/`). Conséquences vérifiées :
- **Tous les numéros de ligne faux** pour l'arbre actif.
- **« 6 stubs qui mentent en retournant OK » : faux** — ce sont des replis
  `#[cfg(not(feature="cuda"))]` qui retournent une **erreur** ; les vraies impls
  (`#[cfg(feature="cuda")]`) appellent libcuda.
- **« Supprimer `VRAMancer/VRAMancer/` » : dangereux et à l'envers** — c'est l'arbre
  de dev **actif**. Détail : `/home/jeremie/VRAMancer/avisclaudeopus.md`.

## B.2 — La ré-analyse `SUPERAUDIT.md` est bonne

Faite sur le bon arbre, numéros vérifiés. DeepSeek **reconnaît son erreur**
(notes revues : complétude 3→6, qualité 5→7, sécurité 4→5) et identifie
correctement `cuda_ffi` / `GpuPipeline` / `GpuNetBridge` / `experimental/` comme
les vraies forces. Audit juste et utile.

## B.3 — Ce que j'ai implémenté (branche `hardening/rust-p0-network`, 3 commits)

Tout ce qui est **sûr + bénéfique + vérifiable** sur cette machine mono-nœud —
**8 items, dont 3 corrigeant les détails de DeepSeek** :

| Item | Détail |
|---|---|
| **P0.1** borne payload | `MAX_PAYLOAD_BYTES` (16 GiB) + garde anti-underflow `total_len<32` + `check_payload_len()` sur les 4 sites de réception (anti OOM-kill distant). |
| **P0.2** timeouts Tokio | 30 s connect / 120 s I/O, 11 `.await` bornés ; `accept()` laissé libre. |
| **P0.3** runtime Tokio global | `shared_runtime()` (OnceLock) ; supprime 4× `Runtime::new()`/appel. |
| **P0.4** garde pointeur cxl | `cxl_direct_memory_{dump,load}` : refus null/0/oversize avant `from_raw_parts`. |
| **P0.5** `direct_vram_load` | Ne fuit plus la VRAM (`mem::forget`+`Ok(0)`) ; échoue franchement (testé live). |
| **P2.8** features Tokio *(corrigé)* | `"full"` → `["net","io-util","rt-multi-thread","sync","time"]`. La liste DeepSeek `["net","macros","sync","time"]` **ne compile pas** (manque io-util + rt-multi-thread). |
| **P2.9** profil release *(corrigé)* | `lto=thin, codegen-units=1, strip=symbols`. **Pas** `panic="abort"` (pyo3 doit capturer les panics, sinon abort tue le process Python). |
| **bonus** | `cudarc` retiré (inutilisé). **Effet mesuré : binaire 1.64 MB → 944 KB (-42 %).** |

Build `--features cuda` OK (3 warnings préexistants), P0 vérifiés live, fixes P0
chargés dans le serveur courant.

## B.4 — Ce que je n'ai PAS fait (vérifié — casserait la prod/les tests)

| Reco | Réalité (importeur réel) |
|---|---|
| P1.6 nettoyer shim `swarm_ledger` | importé par **`core/production_api.py`** (prod) |
| P1.6 shim `vllm_backend` | importé par `server.py:160` |
| P1.6 shim `turboquant` | importé par `tests/test_turboquant.py` |
| P1.5 `webgpu_backend`→experimental | importé par `server.py:848` (POC mais câblé) |
| P3.6 archiver AITP « gelé » | `tests/test_aitp.py` les **teste** (contradiction README↔test) |
| P2.5 XOR 64-bit Rust | la boucle `iter_mut().zip()` **auto-vectorise en AVX-512 à -O3** ; rewrite manuel sans gain sur code de parité critique |

## B.5 — Chantiers restants, re-buckétés selon le matériel RÉEL (corrigé après `reponse_a_opus.md`)

**Correction** : ma formule initiale « aucun validable sur cette machine mono-nœud »
était trop conservatrice — DeepSeek a eu raison de la contester. Matériel confirmé
par l'utilisateur : Desktop **RTX 3090 + RTX 5070 Ti**, Laptop 12e gen **RTX 4060**,
**Mac mini M4 16 Go**, **MacBook Air M5** (+ AMD en recherche). Faits vérifiés
depuis cette session : P2P 3090↔5070 Ti = **NS** (transferts CPU-staged) ; **ReBAR
déjà activé** (BAR1 du 3090 = 32 Go) ; mais le master cluster `192.168.1.100` est
**injoignable maintenant** (swarm hors ligne).

| Bucket | Items | Condition |
|---|---|---|
| **Testable maintenant** (desktop seul) | `P1.3` cuMemGetAddressRange · `P1.4` ReBAR (déjà activé) · `P2.1` RAII CUDA · `P2.2` pool de streams · `P2.6` transfer_async · `P2.7` non-blocking streams · `P2.10` autotune PCIe 3090↔5070 Ti | rien — les 2 GPU locaux suffisent |
| **Testable cluster en ligne** | `P1.1` fenêtre glissante · `P1.2` PipelinedTransport · `P2.3` GpuNetBridge chunked · `P2.4` TLS · `P3.2` pipeline distribué · `P3.8` tests réseau réels | démarrer le swarm (laptop/Mac joignables) |
| **Cross-vendor — préciser l'axe** | bus DMA-BUF NVIDIA↔**AMD** (vrai cross-vendor du `CrossVendorBridge`) ⇒ **besoin du GPU AMD** ; NVIDIA↔**Apple** = axe **réseau** (VTP/Metal-MLX), pas bus PCIe | AMD pour le bus ; Mac en ligne pour le réseau |
| **Vraiment bloqué (matériel absent)** | `P3.1` GPUDirect RDMA (NIC IB/RoCE) · `P3.5` anycast DNS (≥3 nœuds + BIRD) | NIC RDMA / déploiement routé |

## B.6 — Pièges d'environnement découverts (utiles à tous)
- **Deux checkouts** du même repo : parent périmé + nested actif. Ne pas supprimer le nested.
- **Package `.so` shadow** : `maturin develop` ne met pas à jour le module importé
  (un `vramancer_rust/` antérieur intercepte). Remplacer la bonne `.so` à la main.
- **SIGBUS** : `cp` sur une `.so` qu'un process tourne (le serveur charge `vramancer_rust`)
  corrompt son mapping → crash. J'ai ainsi fait tomber le serveur (16:11:33) en
  vérifiant un build ; **relancé, healthy**. Utiliser `mv` (rename atomique) ou
  arrêter le process d'abord.

---

# PARTIE C — Artefacts & état git

**Branches** : `phase7/T7.{0,1,5,6,9}-*` (résultats Phase 7) ·
`phase7/T7.2-boundary-fp8` (note architecte frontière) ·
`hardening/rust-p0-network` (3 commits : P0 + build, off `feat/v6-lending-cooperative`).

**Rapports Phase 7** : `benchmarks/results/phase7/T7.*_report.md` +
`NOTE_ARCHITECTE_boundary_path.md`.

**Échange DeepSeek** : `avisclaudeopus.md` (ma revue + addendum implémentation),
en regard de `conseildeepseek.md` / `avisdeepseek.md` / `SUPERAUDIT.md`.

---

# PARTIE D — Première mesure sur matériel réel (DeepSeek P2.10)

Suite à `reponse_a_opus.md` (DeepSeek : « teste sur le vrai matériel »), première
mesure desktop, **sans rien brancher d'autre** (3090 ↔ 5070 Ti, serveur Qwen3.6
actif). Script : `benchmarks/bench_pcie_bw_3090_5070ti.py` · rapport :
`benchmarks/results/deepseek_p2.10_pcie_bw.md`.

P2P = **NS** entre ces GPU consumer → transferts CPU-staged.

| Transfert | torch naïf | GpuPipeline VRAMancer | gain | chunk optimal |
|---|---|---|---|---|
| 64 MB | 10.4 GB/s | 24.3 GB/s | +134 % | 4 MB |
| 256 MB | 10.4 GB/s | **25.3 GB/s** | **+143 %** | 4 MB |

**Conclusions** :
1. **GpuPipeline validé** : ~25 GB/s effectifs en staged (≈78 % du plafond PCIe 4.0
   x16 d'un saut) → l'overlap DtoH/HtoD du triple-buffering pinned marche. +143 % vs
   `.to()` naïf de torch.
2. **Réponse P2.10** : `chunk_mb = 4` optimal pour cette paire (bat le défaut 16 MB
   de ~+20 %) → cible directe pour un auto-tuner.
3. **Piège de mesure documenté** : `bench_gpu_transfer` expose `bandwidth_gbps`
   (giga-BITS, ×8) ET `bandwidth_gbs` (giga-octets) — nommage à clarifier ; le code
   Rust lui-même est correct (synchronise). (Cette mesure corrige aussi ma B.5 :
   plusieurs items DeepSeek **sont** testables ici — voir tableau re-buckété.)

# Ce qui reste avant « tout fini »
1. **Architecte** : trancher A/B/C sur le cluster frontière (débloque T7.2/3/4 + T7.9-étape2).
2. **Phase 7** : exécuter T7.11 (mesure) puis T7.10 ; T7.7 en dernier.
3. Mettre ce fichier à jour avec ces résultats, puis le figer comme compte-rendu final.
