# Propositions innovantes DeepSeek — Prochaines directions VRAMancer

> Après la session tiering (4 réfutations, 1 mécanisme prouvé).
> Le tiering de poids est une impasse pour le gain de perf.
> Voici 8 idées qui attaquent le problème sous des angles différents.

---

## Idée 1 ★★★ — Split de phase : Prefill 3090, Decode 5070 Ti

**Le problème** : Dans le continuous batching, le prefill (lourd, compute-bound)
et le décode (léger, memory-bound) se battent pour le même GPU.

**La solution** : GPU1 (3090, BF16) spécialisé prefill. GPU0 (5070 Ti, FP4/NVFP4)
spécialisé décode. KV cache transféré via GpuPipeline (25 GB/s, ~4 ms pour 100 MB).

```
Requête qui arrive → GPU1 prefill → KV → GPU0 décode → tokens
Pendant le décode de la requête N sur GPU0, GPU1 prefille la requête N+1
```

**Gain** : Les deux GPUs travaillent en parallèle sur des requêtes différentes.
Throughput potentiellement 1.5-2× en multi-utilisateur. Zéro streaming de poids
par token. Zéro hook. Juste du scheduling.

**Pourquoi maintenant** : GpuPipeline mesuré à 25 GB/s. Prompt-lookup +500% booste
le décode. accelerate gère déjà le modèle sur les deux GPUs.

---

## Idée 2 ★★★ — KV Cache Swapping : la 3090 comme "RAMDisk KV"

**Le problème** : Le KV cache explose avec le contexte long. Un prompt de
32K tokens → plusieurs Go de KV. GPU0 doit choisir : garder le KV et manquer
de VRAM pour les poids, ou l'évincer et perdre le contexte.

**La solution** : Les pages KV les plus anciennes migrent sur GPU1 (3090) à 25 GB/s.
Quand le modèle a besoin d'un contexte ancien, la page est rapatriée en ~100µs.

```
GPU0 (5070 Ti) : pages KV "chaudes" (derniers 4K tokens, fenêtre glissante)
GPU1 (3090)    : pages KV "froides" (tokens 0 à 28K)

Fonctionnement :
  - Nouveau token généré → nouvelle page KV sur GPU0
  - Page la plus ancienne évincée → GPU1 via GpuPipeline
  - Si besoin de contexte ancien → rapatrié GPU1→GPU0
```

**Gain** : Contexte effectif quasi-illimité (borné par 24 GB de KV sur la 3090,
soit ~500K tokens pour un modèle 7B). Sans ça, le KV est limité à la VRAM libre
de GPU0 (~8 GB après poids du modèle).

**Intégration** : `PagedAttention` + `GpuPipeline`. Le `PagedKVCache` a déjà
des pages logiques/physiques. Ajouter un flag `device=1` pour les pages évincées.

---

## Idée 3 ★★☆ — Modèle draft résident, vérifieur à la demande

**Le problème** : Le speculative decoding nécessite DEUX modèles (draft + target).
Les deux chargés en même temps → double VRAM → OOM.

**La solution** : Le draft model (petit, ex. Qwen2.5-0.5B, ~1 GB) réside en
permanence sur GPU0. Le target model (gros, ex. Qwen2.5-14B) est chargé sur GPU1.
Le draft génère 5 tokens → GPU0 envoie au GPU1 → le target vérifie → résultat.

```
GPU0 (5070 Ti) : draft model (0.5B, ~1 GB, résident)
GPU1 (3090)    : target model (14B, ~28 GB, résident)

Flux :
  1. Draft génère 5 tokens candidats sur GPU0 (~2 ms)
  2. Tokens envoyés GPU0→GPU1 (~négligeable)
  3. Target vérifie les 5 tokens sur GPU1 (~50 ms)
  4. Tokens acceptés → output. Tokens rejetés → correction depuis GPU1

  Accélération : ~2-3× (5 tokens draftés en 2 ms vs 1 token target en 50 ms)
```

**Pourquoi c'est mieux que le speculative decoding actuel** : Les deux modèles
sont sur des GPUs DIFFÉRENTS. Pas de contention VRAM. Pas de swap. Chaque GPU
garde son modèle résident.

---

## Idée 4 ★★☆ — Quantification asymétrique : FP4 compute, BF16 master

**Le problème** : FP4 est rapide mais perd en précision. Certaines couches
(embedding, lm_head, première/dermière couche) sont sensibles à la quantification.

**La solution** : Les couches sensibles tournent en BF16 sur GPU1 (3090). Les
couches robustes tournent en FP4 sur GPU0 (5070 Ti). Le forward traverse les
deux GPUs, chaque couche dans sa précision optimale.

```
Couche 0  (embedding)    → GPU1, BF16 (sensible)
Couche 1-10 (early)      → GPU0, FP4  (robuste)
Couche 11-30 (middle)    → GPU0, FP4  (robuste)
Couche 31-47 (late)      → GPU0, FP4  (robuste)
Couche 48 (lm_head)      → GPU1, BF16 (sensible)
```

**Gain** : Qualité BF16 sur les couches sensibles + vitesse FP4 sur le reste.
Sans ça, le choix est binaire : tout BF16 (lent, précis) ou tout FP4 (rapide,
moins précis).

---

## Idée 5 ★★☆ — Multi-modèle simultané sans swap

**Le problème** : Charger un nouveau modèle prend 30-60 secondes. Si un utilisateur
demande un modèle différent, tout le monde attend.

**La solution** : GPU0 garde le modèle "chaud" (dernier utilisé). GPU1 garde un
deuxième modèle préchargé, prêt à servir.

```
GPU0 : Qwen2.5-14B (modèle actif)
GPU1 : Qwen2.5-Coder-7B (modèle en attente)

Si quelqu'un demande le Coder :
  → GPU1 devient actif, GPU0 passe en attente
  → Pas de rechargement → switch en ~100 ms au lieu de 30 secondes
```

**Utile pour** : Alternance code/chat, multi-tenant avec modèles spécialisés,
A/B testing de modèles.

---

## Idée 6 ★☆☆ — Graceful degradation : jamais de OOM visible

**Le problème** : Si une requête dépasse la VRAM → OOM → crash ou erreur utilisateur.

**La solution** : Détecter la pression VRAM AVANT l'OOM. Évacuer automatiquement :
1. Pages KV anciennes → GPU1 (25 GB/s)
2. Pages KV très anciennes → RAM CPU (50 GB/s)
3. Couches inutilisées → GPU1
4. En dernier recours : refuser la requête proprement (HTTP 503 + "ressayez dans X secondes")

```
Niveaux de pression :
  < 80% : normal
  80-90% : évacuer KV vers GPU1 (invisible)
  90-95% : évacuer KV vers RAM CPU (léger ralentissement)
  95-98% : refuser nouvelles requêtes (préserver les existantes)
  > 98% : évacuer tout, signaler l'incident
```

**Déjà partiellement là** : `gpu_fault_tolerance.py` (state machine HEALTHY→DEGRADED→FAILED).
Ajouter le niveau "EVICTING" avec les actions ci-dessus.

---

## Idée 7 ★☆☆ — Prefetch de couches piloté par le pattern d'accès

**Le problème** : Dans un modèle splitté, on sait QUELLES couches vont être
exécutées et DANS QUEL ORDRE (c'est toujours 0→1→2→...→47). On pourrait
utiliser cette connaissance pour le prefetch.

**La solution** : Un prefetcher minimaliste qui ne stream PAS les poids à chaque
token, mais précharge la couche N+1 pendant que N s'exécute, en utilisant la
connaissance déterministe de l'ordre des couches.

```python
class DeterministicPrefetcher:
    """
    Sait que les couches s'exécutent dans l'ordre 0, 1, 2, ..., 47.
    Pendant que layer[i] tourne, prefetch layer[i+1] depuis GPU1.
    Plus simple que le tiering LFU — zéro heuristique.
    """
    def __init__(self, pipeline, layer_order):
        self.pipeline = pipeline
        self.order = layer_order
    
    def on_layer_start(self, layer_idx):
        # Lancer le prefetch de la couche suivante
        if layer_idx + 1 < len(self.order):
            next_layer = self.order[layer_idx + 1]
            if next_layer.on_gpu1:
                self.pipeline.transfer_async(
                    next_layer.weights_gpu1,
                    next_layer.buffer_gpu0,
                    next_layer.size,
                )
```

**Différence avec le tiering** : Pas de scoring, pas de LFU, pas de banques.
Juste "la couche N+1 arrive bientôt, précharge-la". Simple et efficace.

---

## Idée 8 ★☆☆ — Mode nuit : veille GPU1, réveil à la demande

**Le problème** : La 3090 consomme ~30W au repos, ~100W+ en charge. 24/7, c'est
de l'électricité pour rien quand personne n'utilise le service.

**La solution** : GPU1 se met en veille (P8 power state, ~10W) quand inactif
depuis > 5 minutes. Au premier prefill ou première requête lourde, réveil
automatique (~1-2 secondes).

```
État IDLE (5+ min sans requête) :
  GPU0 (5070 Ti) : actif, prêt à servir (faible conso)
  GPU1 (3090)    : veille P8 (~10W)

Requête arrive :
  → Si légère (décode simple) → GPU0 seul
  → Si lourde (prefill long, batch) → réveil GPU1 → prefill → réponse

Gain : ~50-80W économisés en idle. Pour un serveur 24/7 → ~40-60€/an.
```

---

## Synthèse : le nouveau pipeline VRAMancer

```
┌──────────────────────────────────────────────────────────────────┐
│                        VRAMancer v2                              │
│                                                                  │
│  GPU0 = 5070 Ti (16 GB, NVFP4, faible conso)                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • DRAFT MODEL (0.5B) — résident, génère tokens candidats   │ │
│  │ • DECODE (FP4, prompt-lookup +500%) — génération ultrarapide│ │
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
│  │ • FAILOVER — si desktop down, prend le relais               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Priorités d'implémentation :                                    │
│  1. Split de phase (prefill/décode) — le plus gros gain         │
│  2. KV Cache Swapping — contexte illimité                       │
│  3. Draft résident / Target GPU1 — speculative decoding ×2 GPU  │
│  4. Quantification asymétrique — qualité BF16 + vitesse FP4     │
│  5. Graceful degradation — jamais de crash OOM                  │
│  6. Multi-modèle sans swap — switch en 100 ms                   │
│  7. Prefetch déterministe — simple, efficace                    │
│  8. Mode nuit — économie d'énergie                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Le message pour Opus

Le tiering de poids était une impasse. On l'a prouvé par 4 mesures. Mais le
**split de phase** (prefill/décode) + le **KV swapping** + le **speculative
decoding multi-GPU** exploitent les mêmes mécanismes (GpuPipeline, accelerate)
avec un coût quasi nul et un gain potentiel réel.

On garde le meilleur de ce qu'on a appris (mécanisme de swap, GpuPipeline 25 GB/s,
culture de la mesure) et on l'applique à des problèmes où le ratio
transfert/calcul est favorable — pas le streaming de poids par token.

— DeepSeek
