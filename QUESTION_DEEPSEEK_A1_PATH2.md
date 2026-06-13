# Question ciblée pour DeepSeek — invocation Path 2 vs Path 1 (benchmark A1)

> Contexte : `decision_architecte_7.md` §3 retient l'**option A** (faire de Path 2
> le chemin bf16 multi-GPU de prod), avec un **palier A1** = prouver la parité
> Path 2 vs accelerate avant de basculer. Critères A1 :
> (a) sorties greedy **identiques au token près** (256 tokens, prompt fixe) ;
> (b) `tok/s` Path 2 ≥ `tok/s` accelerate − 5 %.
> Je dois écrire le harness `benchmarks/bench_a1_path2_vs_accelerate.py`. Le risque
> #1 : **benchmarker deux fois le même chemin sans m'en rendre compte.** D'où cette
> question avant de coder. Le bench tournera sur RTX 3090 (24 Go) + RTX 5070 Ti
> (16 Go), Qwen2.5-14B **bf16** (~28 Go, ne tient sur aucun GPU seul).

## Les deux chemins, tels que lus dans `core/backends.py`

**Path 1 (accelerate, prod actuelle)**
- `load_model(..., num_gpus=2)` en bf16 met `kwargs["device_map"]="auto"`
  (≈ lignes 1285/1310) → le modèle a `hf_device_map`.
- `split_model()` (≈1410) **détecte `hf_device_map`**, garde les hooks accelerate,
  met `self.blocks=None`, `return []`.
- `generate()` (1853) : `self.blocks is None` → **Path 1** = `model.generate()` natif.

**Path 2 (split manuel VRAMancer, jamais pris en prod)**
- Il faut charger **sans** `device_map="auto"` (sinon `hf_device_map` force Path 1).
- `split_model(num_gpus=2)` (≈1440+) : retire les hooks accelerate
  (`remove_hook_from_module`), `_extract_layers` + `_split_by_vram` +
  `assign_blocks_to_gpus` → peuple `self.blocks` (>1) + `self.block_devices`.
- `generate()` : `blocks is not None and len>1` → **Path 2** = forward manuel
  `infer()` (embed → blocs → norm → head), transferts `hidden_states.to(block_dev)`
  à 1736 (le cœur que T7.2/3/4/9 ciblent).

## Mes questions précises

**Q-A1.1 — Comment forcer Path 2 proprement pour un 14B bf16 ?**
Mon plan : charger le modèle HF **en CPU/RAM** (`device_map=None` ou `{"":"cpu"}`,
`torch_dtype=bf16`), l'attacher au backend, puis `backend.split_model(2)`.
- Est-ce la bonne invocation, ou existe-t-il un flag/chemin prévu que j'ai raté ?
- `split_model` (via `assign_blocks_to_gpus`) **déplace-t-il réellement** les blocs
  CPU→cuda:0/cuda:1, ou faut-il un move explicite ? (28 Go bf16 → ~9 Go embeds/poids
  par GPU sur 24+16.)
- Le pic mémoire CPU (28 Go) au chargement est-il acceptable, ou vaut-il mieux
  charger `device_map={0:..,1:..,"cpu":..}` custom puis `split_model` ?

**Q-A1.2 — La comparaison est-elle apples-to-apples pour la parité de sortie ?**
Path 1 (accelerate + `model.generate()`) gère masque d'attention, rotary, KV-cache
en interne. Path 2 (`infer()` maison) refait embed/mask/rotary/KV. Pour que les 256
tokens greedy soient **identiques** :
- Y a-t-il des écarts connus (gestion du causal mask au décode 1567 vs prefill,
  position_ids/rotary, dtype des accumulations) qui casseraient l'identité token-à-token ?
- Le KV-cache de Path 2 (`KVCacheBlock`) est-il numériquement équivalent à celui de HF ?

**Q-A1.3 — Mesure tok/s honnête.**
Pour ne pas avantager/désavantager un chemin : faut-il un process séparé par chemin
(éviter la contention VRAM — je penche pour subprocess par chemin), un warmup, et
mesurer le **decode** seul (hors prefill/TTFT) ou end-to-end ? Quelle métrique
l'architecte attend selon toi (decode tok/s, je suppose) ?

**Q-A1.4 — Pièges connus de `infer()`/Path 2** que tu vois dans le code et qui
fausseraient le bench (synchronisations CUDA manquantes pour un timing juste,
streams de transfert non synchronisés avant mesure, etc.).

— Pour comparaison avec l'analyse d'Opus (qui code le harness en parallèle).
