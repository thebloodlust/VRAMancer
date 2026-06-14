# Phase 7 — Investigation tiering & pivot (juin 2026)

> **Synthèse unique** de la session Opus × DeepSeek × architecte sur le tiering.
> Les échanges détaillés (≈46 fichiers `reponse_*`, `QUESTION_*`, `decision_architecte_*`,
> `resultat_*`) sont archivés dans ce dossier. Ce fichier suffit à comprendre l'essentiel.
> Date : 2026-06-14. Branche : `phase7/A1-parity`.

---

## 1. Ce qu'on a testé, ce qu'on a mesuré

L'idée de départ (vision initiale du projet) : **un GPU calcule, l'autre prête sa
VRAM comme magasin de poids** (« le modèle n'y voit que du feu »). On l'a explorée
de bout en bout, **tout mesuré**.

| Hypothèse | Prédiction | Mesure | Verdict |
|---|---|---|---|
| A1 — bug = masque attention (DeepSeek) | « fix trivial » | no-op | ❌ faux |
| A1 — bug = transfert sync (Opus) | « fix sync » | no-op, reproduit sur 1 GPU | ❌ faux → vrai bug = `cache_position` absent dans `infer()` |
| Tiering dense via **GpuPipeline** (Rust 25 GB/s) | « > 90% du ref » | **61%** | ❌ overhead par appel domine |
| Tiering dense **packé** v0.3 | « 80-85% » | **64%** | ❌ torch.copy_ reste meilleur |
| Tiering dense **torch double-buffer** v0.1 | — | **73.1%** | ✅ meilleure variante (mais ~27% de coût) |
| **MoE-tiering** (chaud/froid experts) | « LE différenciant » | distribution **uniforme** | ❌ **réfuté** |

**Mesures clés :**
- Dense tiering : meilleur cas **73.1%** du débit de référence, **transfert-bound**,
  ~27% de coût irréductible. **accelerate `device_map="auto"` fait mieux** (5.41 tok/s
  sur le 14B, sans swap, en pipeline-parallèle).
- MoE (`benchmarks/probe_expert_usage.py`, Qwen1.5-MoE-A2.7B, 60 experts, top_k=4) :
  couverture top-8 = **15.3%** (uniforme = 13.3%), prefill active **47.7/60 experts (80%)**.
  → distribution **quasi-uniforme PAR CONCEPTION** (perte de load-balancing à
  l'entraînement) → **pas de chauds/froids** → streamer les experts actifs n'économise rien.

## 2. La conclusion honnête (sans hype)

**Le tiering de poids n'est ni un gain de perf ni un différenciant fort.**
- Dense : accelerate fait mieux.
- MoE : réfuté (load-balancing).

**Valeur honnête restante du tiering** : seulement « faire tenir un modèle qui dépasse
la VRAM combinée des GPU » (modeste, vs offload CPU — le lending pool le faisait déjà)
+ **cross-vendor** (NVIDIA↔AMD, non prouvé, nécessite un GPU AMD).

**La vraie valeur de VRAMancer reste** : orchestration multi-GPU hétérogène **via
accelerate/llama.cpp** (pas un moteur maison) + **optimisations mesurées**
(prompt-lookup +500%, TurboQuant, DirectFP4, lending, auto-heal). À refléter dans le
README — déjà fait (PR #7). Pas de « 0→X », « révolutionnaire », « rend l'impossible
possible ».

## 3. La discipline (le vrai acquis)

**4 fois cette session, la mesure a corrigé l'intuition** (A1, GpuPipeline, packing,
MoE). Mesurer d'abord a évité de coder ~150 lignes de streaming d'experts inutile.
**4 « non » mesurés valent mieux que 4 « oui » imaginés.** C'est la méthode du projet.

## 4. Le pivot (convergence 3 cerveaux)

Puisque le streaming de poids est une impasse (on retransfère le même poids à chaque
token — ratio transfert/calcul défavorable), le bon angle est :

> **Deux GPU hétérogènes sur des tâches COMPLÉMENTAIRES**, pas du streaming de poids.

DeepSeek a proposé 18 idées (8 techniques + 10 stratégiques, voir `DEEPSEEK_COMPLET.md`).
Filtrées avec la même discipline. **3 caveats matériels** identifiés :

1. **Split prefill/décode** : exige le modèle entier résident sur **chaque** GPU
   (14B BF16=28 Go ne tient sur aucun seul → fenêtre réelle ≤12B) ; gain 1.5-2×
   **multi-user uniquement** (single-stream ≈ 0). → à **mesurer** avant de croire.
2. **KV swapping chaud/froid intra-génération** : **même piège que le tiering**
   (l'attention lit toutes les pages à chaque token). Version honnête = offload
   **inter-session** (tenants en pause).
3. **Spec decode ×2 GPU** : le gain 2-3× vient du spec decode lui-même, pas du split
   (le split *permet* juste de loger draft+target).

## 5. Plan validé (priorité)

| # | Chantier | Statut |
|---|---|---|
| 1 | **S1 — `vramancer.patch()` drop-in** (packaging des optims prouvées + max_memory anti-OOM) | à faire — **sûr, zéro recherche GPU** |
| 2 | **S2 — `vramancer quickstart <use-case>`** (UX sur l'existant) | à faire après S1 |
| 3 | **UNE mesure disagg** prefill/décode (7-12B, multi-user) | à mesurer avant de creuser |
| 4 | Tout le reste (KV inter-session, spec decode, FP4 asym…) | backlog, mesuré au cas par cas |

**Décision design S1 (user)** : implémenter **les deux variantes** (légère = monkeypatch
fin ; lourde = via `InferencePipeline`), **les tester**, garder celle qui a des résultats.

## 6. Artefacts (code, dans `benchmarks/`)

- `probe_expert_usage.py` — probe distribution experts MoE (la mesure qui réfute le MoE-tiering).
- `tiering_v0_1_doublebuffer.py` — meilleure variante dense (73.1%, correcte, VRAM-efficient).
- `tiering_v0_2_gpupipeline.py` / `tiering_v0_3_packed.py` — GpuPipeline réfuté.
- `tiering_v0_prefetch.py` / `poc_tiering_offload_gpu1.py` — POC / v0.
- `test_a1_single_gpu.py` / `test_a1_accelerate_baseline.py` / `bench_a1_path2_vs_accelerate.py` — harness A1.

## 7. Pour l'architecte (« fable »)

Docs de décision archivés ici : `decision_architecte_7.md`, `decision_architecte_8.md`,
`ETAT_EXECUTION_DECISION_ARCHITECTE_7.md`, `COMPTE_RENDU_ARCHITECTE_PHASE7_DEEPSEEK.md`.
Le doc complet DeepSeek (pivot + 18 idées) : `DEEPSEEK_COMPLET.md`. Le filtre Opus :
`reponse_opus_sur_deepseek_complet.md`. L'accord final : `reponse_deepseek_finale.md`.
