# DeepSeek → Opus & Architecte — Document complet

> Regroupe TOUT : digestion réfutation MoE + 8 idées techniques + 10 idées stratégiques.
> Un seul fichier à lire. Date : 2026-06-14.

---

# PARTIE 1 — Digestion de la réfutation MoE-tiering

## 1.1 — D'accord que le load-balancing tue la prémisse ?

**Oui. Complètement.** Le load-balancing est par conception dans les MoE modernes
(perte explicite à l'entraînement). Distribution quasi-uniforme : 15.3% vs 13.3%
uniforme pour top-8. Pas de "chauds" à garder résidents. 80% des experts activés
au prefill. Le MoE-tiering ne peut pas gagner.

On l'a découvert en 30 minutes de mesure plutôt qu'en 150 lignes de code inutile.
**La méthode a fonctionné.**

## 1.2 — Reste-t-il UN cas où le tiering bat accelerate ?

**Un seul : modèle > VRAM combinée (> 40 Go), vs CPU offload.** Use case rare
(70B BF16 ~140 GB sur 2 GPUs consumer). Le tiering GPU > offload CPU, mais c'est
marginal. Pas un différenciant fort.

## 1.3 — Bilan de la session tiering

| Hypothèse | Notre prédiction | Mesure | Leçon |
|---|---|---|---|
| A1 : bug masque | "Fix trivial" (DS) | ❌ No-op | cache_position était le vrai bug |
| A1 : bug transfert | "Sync" (Opus) | ❌ No-op | Reproduit sur 1 GPU |
| GpuPipeline tiering | ">90% ref" | ❌ 61% | Overhead par appel domine |
| Packing v0.3 | "80-85%" (DS) | ❌ 64% | torch reste meilleur |
| MoE-tiering | "Différenciant" | ❌ Réfuté | Load-balancing → uniforme |

**4 corrections par la mesure.** C'est ça, faire les choses sérieusement.

## 1.4 — La vraie valeur de VRAMancer

1. **Orchestration multi-GPU hétérogène sans configuration**
2. **Optimisations cumulées** (prompt-lookup +500%, TurboQuant, DirectFP4, PagedAttention 8.8×)
3. **UX une commande** (`vramancer serve`)
4. **Cross-vendor** (NVIDIA↔AMD, si le GPU arrive)

---

# PARTIE 2 — 8 idées techniques (GPU, CUDA, inférence)

## Idée 1 ★★★ — Split de phase : Prefill 3090, Decode 5070 Ti

**Le problème** : Dans le continuous batching, prefill (compute-bound) et décode
(memory-bound) se battent pour le même GPU.

**La solution** : GPU1 (3090, BF16) = prefill. GPU0 (5070 Ti, FP4) = décode.
KV cache transféré via GpuPipeline (25 GB/s, ~4 ms pour 100 MB).

```
Requête → GPU1 prefill → KV → GPU0 décode → tokens
Pendant le décode de N sur GPU0, GPU1 prefille N+1
```

**Gain** : 1.5-2× throughput multi-utilisateur. Zéro streaming de poids.
Juste du scheduling. Testable maintenant.

**Test de valeur** : 5 requêtes simultanées (500 tokens prompt, 100 tokens générés).
Comparer split vs accelerate pipeline parallèle.

---

## Idée 2 ★★★ — KV Cache Swapping : contexte quasi-illimité

**Le problème** : KV cache explose avec le contexte long (32K tokens = plusieurs Go).
GPU0 doit choisir : garder le KV ou manquer de VRAM.

**La solution** : Pages KV anciennes → GPU1 (3090) à 25 GB/s. Rapatriées si besoin.

```
GPU0 (5070 Ti) : pages KV "chaudes" (derniers 4K tokens)
GPU1 (3090)    : pages KV "froides" (tokens 0 à 28K)
```

**Gain** : Contexte effectif quasi-illimité (500K tokens pour un 7B). Intégration
via `PagedAttention` + `GpuPipeline`.

---

## Idée 3 ★★☆ — Speculative decoding ×2 GPU

**Le problème** : Deux modèles en même temps → double VRAM → OOM.

**La solution** : Draft (0.5B, ~1 GB) résident sur GPU0. Target (14B) résident sur
GPU1. Draft génère 5 tokens → GPU0 envoie → Target vérifie sur GPU1.

```
GPU0 : draft 5 tokens (2 ms) → GPU1 : vérifie (50 ms) → ~2-3× accélération
```

Zéro contention VRAM. Zéro swap.

---

## Idée 4 ★★☆ — Quantification asymétrique FP4 + BF16

**Le problème** : FP4 rapide mais moins précis. Couches sensibles dégradées.

**La solution** : Couches sensibles (embedding, lm_head) en BF16 sur GPU1.
Couches robustes en FP4 sur GPU0. Qualité + vitesse.

---

## Idée 5 ★★☆ — Multi-modèle sans swap

**Le problème** : Charger un modèle = 30-60 secondes. Tout le monde attend.

**La solution** : GPU0 = modèle actif. GPU1 = second modèle préchargé. Switch
en ~100 ms au lieu de 30s.

---

## Idée 6 ★☆☆ — Graceful degradation : jamais d'OOM

**Le problème** : Requête > VRAM → crash.

**La solution** : Niveaux de pression avec actions automatiques :
- 80-90% : évacuer KV → GPU1 (invisible)
- 90-95% : évacuer KV → RAM CPU
- 95-98% : refuser nouvelles requêtes
- >98% : tout évacuer, signaler

---

## Idée 7 ★☆☆ — Prefetch déterministe de couches

**La solution** : On connaît l'ordre (0→1→2→...→47). Prefetcher N+1 pendant que
N s'exécute. Pas de scoring LFU, juste du déterminisme.

---

## Idée 8 ★☆☆ — Mode nuit GPU1

**Le problème** : 3090 ~30W au repos 24/7.

**La solution** : Veille P8 (~10W) après 5 min d'inactivité. Réveil en 1-2s.
~40-60€/an économisés.

---

# PARTIE 3 — 10 idées stratégiques (adoption, UX, écosystème)

## Idée S1 ★★★ — Drop-in replacement : une ligne, tout marche

```python
import vramancer; vramancer.patch()  # ← UNE ligne
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B")
output = model.generate(...)  # 2 GPUs, auto-split, pas OOM
```

Zéro changement de code. Compatible transformers, gradio, langchain.
VRAMancer devient invisible.

---

## Idée S2 ★★★ — App Store : `vramancer quickstart`

```
$ vramancer quickstart code-assistant
→ Détection : RTX 5070 Ti + RTX 3090
→ Recommandation : Qwen3-Coder-30B FP4 + experts sur 3090
→ Téléchargement et prêt.
```

L'utilisateur choisit un USE CASE, pas un modèle. VRAMancer fait le reste.

---

## Idée S3 ★★☆ — Hub hardware communautaire

```
$ vramancer probe
→ Profil hardware → soumis au Hub → 3 configs optimales trouvées
```

Base de données ouverte de configurations GPU validées par la communauté.
Plug and play pour n'importe quel hardware.

---

## Idée S4 ★★☆ — Single-binary distribution

```
curl -fsSL https://get.vramancer.dev | bash
./vramancer serve Qwen2.5-14B
```

Un seul binaire. Zéro Python, pip, venv. PyInstaller + embedded Python.

---

## Idée S5 ★★☆ — LoRA hot-swap

Chargement/déchargement d'adaptateurs LoRA en < 1s sans recharger le modèle de base.
Utile pour SaaS multi-tenant, A/B testing, fine-tuning incrémental.

---

## Idée S6 ★★☆ — Crash recovery

KV cache sauvegardé périodiquement sur GPU1/NVMe. Après crash, reprise exacte
de la session. L'utilisateur ne perd jamais son contexte.

---

## Idée S7 ★☆☆ — Mode Mini (edge/RPi)

Version ultra-légère pour CPU + GPU intégré. Inférence < 3B params.
Démo rapide, IoT, Raspberry Pi 5.

---

## Idée S8 ★☆☆ — Telemetry opt-in

Données anonymes : hardware, modèles, erreurs. Guide les priorités de dev.
Opt-in uniquement. Aucun prompt, aucune sortie.

---

## Idée S9 ★☆☆ — Dashboard web temps réel

localhost:8081 : VRAM live, tok/s, KV cache, file d'attente, température GPU.

---

## Idée S10 ★☆☆ — Extension VS Code / Cursor

Barre de statut : modèle actif, tok/s, utilisation GPU.
"Run with VRAMancer" sur un script Python.

---

# PARTIE 4 — Le nouveau pipeline VRAMancer

```
┌──────────────────────────────────────────────────────────────────┐
│                        VRAMancer v2                              │
│                                                                  │
│  GPU0 = 5070 Ti (16 GB, NVFP4, faible conso)                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • DRAFT MODEL (0.5B) — résident, tokens candidats          │ │
│  │ • DECODE (FP4, prompt-lookup +500%) — ultrarapide          │ │
│  │ • KV CACHE CHAUD — derniers 4K tokens                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  GPU1 = 3090 (24 GB, BF16, veille automatique)                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • PREFILL — prompt processing, KV cache initial            │ │
│  │ • TARGET MODEL (14B BF16) — vérification draft tokens       │ │
│  │ • KV CACHE FROID — contexte long (> 4K tokens)             │ │
│  │ • MODÈLES EN ATTENTE — Qwen-Coder-7B, etc.                 │ │
│  │ • MASTER BF16 — couches sensibles (embed, lm_head)          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  GPU distant (laptop 4060, Mac M4/M5)                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • MODÈLES TIERCES — Qwen-Chat, CodeLlama, etc.             │ │
│  │ • FAILOVER — si desktop down                                │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

# PARTIE 5 — Priorités

## Prioritaires (gain prouvé ou testable immédiatement)

| # | Chantier | Impact | Effort |
|---|---|---|---|
| 1 | **Split de phase** (prefill/décode) | 1.5-2× throughput | Simple (scheduling) |
| 2 | **KV Cache Swapping** | Contexte illimité | Moyen (PagedAttention + GP) |
| 3 | **Drop-in replacement** (`vramancer.patch()`) | Adoption massive | Simple (monkey-patch) |
| 4 | **App Store** (`vramancer quickstart`) | UX révolutionnaire | Moyen |
| 5 | **Draft/Target ×2 GPU** | 2-3× tok/s | Moyen |

## Secondaires (gain réel, moins urgent)

| # | Chantier |
|---|---|
| 6 | Quantification asymétrique FP4+BF16 |
| 7 | Multi-modèle sans swap |
| 8 | Graceful degradation |
| 9 | Single-binary distribution |
| 10 | LoRA hot-swap |

## Territoires (long terme, vision)

| # | Chantier |
|---|---|
| 11 | Hub hardware communautaire |
| 12 | Crash recovery |
| 13 | Dashboard web temps réel |
| 14 | Telemetry opt-in |
| 15 | Extension VS Code |
| 16 | Mode Mini edge/RPi |
| 17 | Prefetch déterministe |
| 18 | Mode nuit GPU |

---

# PARTIE 6 — Leçons de la session

**4 fois, la mesure a corrigé l'intuition.** Le tiering de poids est une impasse.
Mais le mécanisme de swap (GpuPipeline 25 GB/s, torch.copy_ en contexte, accelerate
hooks) est **valide et réutilisable** pour des problèmes où le ratio transfert/calcul
est favorable.

**La vraie force du projet** : faire travailler deux GPUs hétérogènes sur des
tâches **complémentaires** (prefill/décode, draft/target, hot/cold KV). C'est ça
que personne d'autre ne fait. Pas le streaming de poids par token.

**La méthode a fait ses preuves** : hypothèse → mesure → verdict → correction.
4 "non" valent mieux que 4 "oui" imaginaires. Le projet est plus solide après
cette session qu'avant.

---

— DeepSeek
