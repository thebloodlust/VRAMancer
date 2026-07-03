# S1 — `vramancer.patch()` drop-in : les 2 variantes testées, LIGHT gagne

> Méthode du projet : on teste les deux, on garde celle qui a des résultats.
> Mesure `benchmarks/test_s1_patch.py` sur TinyLlama-1.1B-Chat, 2 GPU (3090 + 5070 Ti).
> Date : 2026-06-14.

## Ce que `patch()` fait (honnête)
Le split multi-GPU **est** accelerate. `patch()` n'invente pas de moteur — il package
en une ligne ce qu'on a **mesuré** : `device_map="auto"` + `max_memory` compute-aware
(anti-OOM), `expandable_segments`, et prompt-lookup par défaut dans `.generate`.

## Résultat comparatif

| Critère | **LIGHT** (monkeypatch fin) | HEAVY (via InferencePipeline) |
|---|---|---|
| Marche | ✅ | ✅ (seulement après fix récursion) |
| load / gen (TinyLlama) | 2.26 s / 1.63 s | 1.47 s / 1.12 s |
| Reste un **modèle HF** | ✅ oui (`is_hf_module=True`) | ❌ objet custom |
| API | HF-native (tokenizer + `generate`) | `.generate(prompt:str)->str` non-HF |
| Injection vérifiée | `device_map=auto`, `max_memory={0:22.4,1:14.7,cpu:48}`, prompt-lookup actif | n/a |
| Fragilité | aucune | récursion (le backend rappelle `from_pretrained`) ; backend `auto`→vLLM crashe sur petit modèle |

## Verdict : LIGHT

Le HEAVY est un poil plus rapide **ici**, mais :
1. C'est un **artefact** : le LIGHT force `device_map="auto"` (2 GPU) sur un modèle qui
   tient sur 1 GPU → léger surcoût de dispatch. Sur un **gros** modèle (la cible réelle
   de VRAMancer), cet écart disparaît. *(Amélioration possible : placement mono-GPU si
   le modèle tient — comme le fait déjà la pipeline ; non critique car la cible = gros modèles.)*
2. Le HEAVY **casse la compat HF** (retourne un objet non-`nn.Module` → plus de gradio,
   langchain, `.forward`, training, `.to()`…). Or **rester compatible HF est tout
   l'intérêt d'un drop-in**. Le HEAVY le détruit.
3. Le HEAVY a une **fragilité structurelle** : monkeypatcher `from_pretrained` alors que
   le backend de la pipeline l'appelle lui-même → récursion infinie (corrigée ici en
   restaurant l'original pendant le load, mais c'est un symptôme).

→ **`vramancer.patch()` = variante légère.** Le mode lourd reste accessible via
`patch("heavy")` mais documenté **expérimental / non-HF**.

## Code
- `vramancer/dropin.py` — `patch()`, `unpatch()`, `is_patched()`, `compute_max_memory()`.
- `vramancer/__init__.py` — expose l'API publique.
- `benchmarks/test_s1_patch.py` — le harnais comparatif.

## Suite
- (Option) placement mono-GPU dans le light si le modèle tient (gomme le surcoût).
- S2 `vramancer quickstart <use-case>` (UX sur l'existant).
- LA mesure disagg prefill/décode (7-12B, multi-user) avant de creuser le pivot GPU.
