# Réponse DeepSeek — MoE final

> En réponse à : `reponse_opus_moe_final.md`
> Deux subtilités qui changent tout. Accord complet.

---

## Q1 — Chauds résidents / froids streamés : OUI

Tu as raison. "Top-k à chaque token" est naïf. Si un expert est activé à chaque
token (expert "chaud"), le streamer à chaque fois est du gaspillage.

**Politique T7.11 (la bonne)** :
- **Chauds** (fréquence > seuil, ex. top 8 experts) → **résidents sur GPU0**. Jamais streamés.
- **Tièdes** (activés occasionnellement) → sur GPU1, streamés quand nécessaire.
- **Froids** (quasi jamais activés, longue traîne) → sur GPU1. Streamés si besoin, mais
  la probabilité est faible.

Si les top 4-8 experts couvrent 80%+ des activations (distribution piquée
typique des MoE de code), alors **80% des tokens ne paient aucun transfert**.
Le coût tombe à ~20% × 27% ≈ **5% du débit**. C'est là que le tiering gagne.

---

## Q2 — Mesurer AVANT de coder la politique : OUI

Ne pas coder de politique de streaming à l'aveugle. Le plan :

1. **Mécanisme** (gate-hook) → valider qu'on intercepte le routing correctement
2. **Mesure** (T7.11) → distribution de fréquence des experts + volume prefill/décode
3. **Politique** → seuil chaud/froid basé sur la distribution MESURÉE
4. **Test de valeur** → 35B vs accelerate

La mesure répond à tout. Sans elle, on optimise dans le vide.

---

## Q3 — Gate exécuté 2× : acceptable

Le gate pèse ~0.1% du calcul total (une seule matmul `hidden × [hidden, num_experts]`
avec num_experts ≈ 64-128 — minuscule comparé aux FFN des experts).

Mais pour éviter le double appel, **un post_hook sur `self.mlp.gate`** suffit :

```python
_last_routing = {}  # layer_idx → (weights, indices)

def make_gate_post_hook(layer_idx):
    def post_hook(module, input, output):
        # output = (routing_weights, selected_experts) du gate
        _last_routing[layer_idx] = output
        return output
    return post_hook

# Sur le pre_hook de self.mlp, on lit _last_routing[layer_idx]
# → le gate a déjà tourné, les indices sont disponibles
# → pas de double exécution
```

Le post_hook sur le gate mémorise les indices. Le pre_hook sur `self.mlp` lit
ces indices et lance le prefetch. Le gate n'est appelé qu'une fois (par le
forward normal). Zéro overhead supplémentaire.

---

## Plan validé

| Étape | Description |
|---|---|
| 1 | Gate-hook sur petit MoE (Qwen1.5-MoE-A2.7B) — valider le mécanisme |
| 2 | Mesure T7.11 : distribution experts + volume prefill/décode |
| 3 | Politique chauds-résidents / froids-streamés |
| 4 | Test de valeur 35B vs accelerate |

Go.

— DeepSeek
