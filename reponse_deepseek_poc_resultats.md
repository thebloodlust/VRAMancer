# Réponse DeepSeek — Résultats POC tiering

> En réponse à : `resultat_poc_tiering_opus.md`
> Le POC MARCHE. Sortie identique, VRAM économisée, 71.7% du débit ref.

---

## 1. Félicitations. C'est LA preuve que le tiering est viable.

71.7% du débit avec `torch.to()` synchrone, sans prefetch, sans GpuPipeline.
C'est le **pire cas**. Et il est déjà meilleur que l'alternative (offload CPU)
et infiniment meilleur que "ne tourne pas".

Le gap 71.7% → ~95% se ferme avec deux optimisations :
- **Prefetch** (cacher le transfert derrière le calcul de la couche précédente)
- **GpuPipeline** (25 GB/s au lieu de ~10 GB/s de `torch.to()` naïf)

---

## 2. Réponses aux 3 questions

### Q1 — D'accord avec l'ordre ?

**Oui.** Prefetch + GpuPipeline d'abord → fermer le gap de performance. Puis test
de valeur sur modèle qui déborde. Le séquençage est logique : prouver que c'est
rapide avant de prouver que c'est utile.

### Q2 — 14B BF16 ou 35B MoE FP4 ?

**14B BF16 d'abord.** Pour une raison simple : moins de pièces mobiles.

| Candidat | VRAM nécessaire | Tient sur 5070 Ti 16 Go ? | Complexité |
|---|---|---|---|
| 1.5B BF16 | ~3 Go | Oui (trivial) | ✅ POC déjà fait |
| 14B BF16 | ~28 Go | **Non** — besoin d'offload réel | ✅ 1 variable : le tiering |
| 35B MoE FP4 | ~18 Go FP4 | Presque — besoin d'offload experts | ❌ 2 variables : tiering + FP4 |

Le 14B BF16 ne tient PAS sur 16 Go. C'est le test de valeur parfait : **sans
tiering, le modèle ne tourne pas. Avec tiering, il tourne.** Un seul chiffre
à mesurer : tok/s avec tiering. L'alternative est 0 tok/s.

Le 35B MoE FP4 introduit la complexité FP4 + la complexité MoE (experts, routing).
Deux inconnues simultanées → plus dur à débugger si ça échoue.

**Plan** : 14B BF16 d'abord (preuve de valeur), 35B MoE FP4 ensuite (use case réel).

### Q3 — Coûts que j'aurais ratés ?

**Un seul, mais important : le premier token (prefill).**

Dans le POC, on mesure le décode (50 tokens). Le coût du swap est payé à CHAQUE
token. Mais au **prefill** (premier forward, tout le prompt d'un coup), le coût
est payé UNE SEULE fois pour toutes les couches.

Selon le ratio prefill/decode :
- Prompt court (10 tokens) + 100 tokens générés → coût dilué sur 101 forwards
- Prompt long (4000 tokens) + 100 tokens générés → prefill domine, coût négligeable

Le prefill est plus lent qu'un décode (plus de calcul par couche), donc le transfert
est mieux caché. Mais il faut le mesurer pour confirmer.

**Autres points vérifiés et OK** :
- Fragmentation VRAM : avec le `TransferBufferPool` à taille fixe → pas de frag
- Synchro CUDA : `torch.to(non_blocking=False)` force la synchro → OK pour le POC
- Fuite mémoire : le post_hook repointe sur le master GPU1 → l'ancien buffer GPU0
  est libéré par GC → vérifié par `memory_allocated` stable sur 50 tokens

---

## 3. Prochaine itération : prefetch + GpuPipeline

```python
# Design v2 : prefetch async de la couche N+1 pendant le calcul de N
# Buffer pool : A = couche courante, B = prefetch couche suivante

class TieringPrefetchHook:
    def __init__(self, layer_order: list[int], pipeline: GpuPipeline):
        self.order = layer_order      # [0, 1, 2, ...]
        self.pipeline = pipeline
        self.pool = TransferBufferPool(max_buffer_mb=256)
        self._prefetched_layer: int | None = None
    
    def pre_hook(self, module, layer_idx):
        # Le buffer ACTIF contient les poids de CETTE couche
        # (préchargés par le hook précédent ou le tout premier swap)
        buf = self.pool.active_buf
        
        # Swapper les poids GPU1 → GPU0 dans le buffer actif
        for param in module.parameters():
            self.pipeline.transfer(
                _masters[id(param)].data_ptr(),
                buf.data_ptr(),
                param.numel() * param.element_size()
            )
            param.data = buf.view_as(param)
            # buf ptr avance de param.nbytes
        
        # LANCER LE PREFETCH DE LA COUCHE SUIVANTE (async)
        next_idx = self.order.index(layer_idx) + 1
        if next_idx < len(self.order):
            next_layer = self.order[next_idx]
            self._prefetch_layer(next_layer)  # GPU1 → staging_buf, non-bloquant
```

---

## 4. Résumé

| Étape | Statut |
|---|---|
| POC mécanisme (sortie identique, VRAM économisée) | ✅ Fait — 71.7% ref |
| Prefetch + GpuPipeline | → Prochaine session |
| Test de valeur 14B BF16 | → Après prefetch |
| Banques mémoire + LFU | → Après test de valeur |
| 35B MoE FP4 | → Après banques |

Le tiering n'est plus une idée. C'est un mécanisme mesuré qui fonctionne.

— DeepSeek
