# experimental/manual_forward — forward manuel VRAMancer (Path 2), DÉPRÉCIÉ post-A1

> Statut : **déprécié**. N'est PAS le chemin de production. Conservé pour référence
> (règle R1 : rien n'est supprimé). Décision architecte 2026-07-03, palier A1.

## Ce que c'est
Le « Path 2 » : le forward manuel de VRAMancer qui découpe le modèle en blocs
(`KVCacheBlock`) et exécute `embed → blocs (avec KV-cache) → norm → lm_head` à la main,
avec transferts inter-GPU. L'idée était de posséder le pipeline plutôt que déléguer à
`accelerate`.

## Pourquoi c'est déprécié (verdict A1)
Le palier A1 (`RESULTAT_PALIER_A1.md`) a mesuré que ce forward manuel est **cassé** :
sortie dégénérée (« 1. 1. 1. »), **prouvé sur un seul GPU** (donc bug logique, pas
multi-GPU). **Cause racine** : `cache_position` n'est pas propagé aux couches. Depuis
transformers ≥ 4.45, sans `cache_position` le `DynamicCache` est tissé mais **jamais
peuplé** → le modèle ré-attend à vide → dégénérescence.

Et **on ne le répare pas** : la mesure de session montre que même réparé, le split manuel
ne battrait pas `accelerate` (régime décode-dominé, transfert-bound, P2P indisponible/217).
Le +143 % d'avantage transfert du GpuPipeline ne renverse pas un régime dominé par le décode.

## Où vit le code
Le code réel reste dans `core/backends.py` (méthode `_infer_with_kv_cache` /
`__infer_with_kv_cache_impl`, classe `KVCacheBlock`) car ce sont des méthodes couplées à
`HuggingFaceBackend` (extraction physique risquée). Elles sont **marquées dépréciées** et
émettent un `DeprecationWarning` à l'usage. Le chemin de prod (`self.blocks is None` →
`accelerate device_map` → `model.generate`) n'est pas concerné.

## Chemin de production à utiliser
`accelerate` (`device_map="auto"`) pour bf16 multi-GPU, `llama.cpp` pour GGUF. VRAMancer
orchestre au-dessus. Cf. `BENCHMARK_RESULTS.md` (5.41 tok/s mesuré) et le README.

## Critères de repromotion (si un jour)
1. `cache_position` propagé + A1 repasse (sorties identiques à accelerate au token près) ;
2. ET un gain mesuré vs accelerate sur un régime réel (aujourd'hui : aucun). 
Tant que (2) n'est pas prouvé, ne pas ré-investir.
