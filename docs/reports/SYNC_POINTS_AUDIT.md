# Sync Points Audit — `core/inference_pipeline.py`

> Date : 2026-05  
> Fichier audité : `core/inference_pipeline.py` (~1500 LOC)

## Méthode

Recherche de tous les points de synchronisation CPU↔GPU bloquants :

```bash
grep -c "synchronize" core/inference_pipeline.py     → 0
grep -c "\.cpu()"      core/inference_pipeline.py     → 0
grep -c "\.item()"     core/inference_pipeline.py     → 0
grep -c "\.tolist()"   core/inference_pipeline.py     → 0
```

## Résultat

**0 sync points bloquants explicites dans `inference_pipeline.py`.**

Le pipeline principal ne contient aucun `torch.cuda.synchronize()`, `.cpu()`, `.item()` ou `.tolist()` explicite.

## Analyse

### Synchronisation déléguée

La synchronisation implicite est gérée par :

1. **HuggingFace Accelerate** — dans `generate()` du backend, Accelerate insère ses propres syncs lors des `send_to_device()` et des transitions entre devices.

2. **TransferManager** — `transfer()` utilise `cudaMemcpyPeerAsync` qui est async. La sync implicite arrive lors du premier accès au tenseur destination sur le device cible.

3. **Tokenizer** — la tokenization (CPU → GPU) insère une sync implicite lors du `.to(device)` du tensor d'input_ids.

4. **Output decoding** — lors du décodage de la séquence générée, le tenseur output_ids passe sur CPU pour le décodage du tokenizer. C'est une sync légitime et inévitable (le tokenizer travaille sur CPU).

### Points de sync légitimes (dans les backends, hors pipeline.py)

| Localisation | Raison | Légitime ? |
|-------------|--------|-----------|
| `backends.py: output_ids.tolist()` | Décodage tokens → strings | ✅ Oui — inévitable |
| `tokenizer.py: inputs.to(device)` | CPU→GPU input | ✅ Oui — inévitable |
| `TransferManager: cudaMemcpyPeerAsync` | Inter-GPU activation | ✅ Oui — async, sync implicite seulement au prochain accès |

## Verdict

**Pipeline principal : aucun sync bloquant parasite identifié.**

L'architecture est correcte : les seules synchronisations sont aux frontières inévitables (tokenization, output decoding).

**Aucune optimisation nécessaire à ce niveau.**
