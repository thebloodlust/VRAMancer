# C7 — brouillon de lancement (à publier par Jérémie)

> Prep par Opus. La publication (PyPI, Reddit) reste manuelle (comptes/identité).
> Prérequis avant post : PyPI `pip install vramancer` OK sur machine vierge + GIF asciinema.

## Post r/LocalLLaMA (brouillon)

**Titre** : Local coding agent on mismatched consumer GPUs (RTX 3090 + RTX 5070 Ti):
Qwen3.6-35B-A3B with OpenAI-compatible tool calling, one command

**Corps** :
> I've been building VRAMancer — an orchestrator that runs big models across **mismatched**
> consumer GPUs (different generations, no NVLink) and exposes an **OpenAI-compatible API
> with working function calling**, so tools like Aider/Cline just work against a local model.
>
> On a 3090 + 5070 Ti (Ampere + Blackwell, P2P blocked), it serves Qwen3.6-35B-A3B (GGUF)
> and I validated a full **Aider session end-to-end**: it edits code and updates the tests,
> unassisted. The tool-call round-trip (call → result → final answer) is handled server-side.
>
> One command: `./serve_qwen36.sh` → point your agent at `localhost:5030/v1`.
>
> Honest about what it is: it's an **orchestration + UX layer on top of accelerate/llama.cpp**,
> not a new inference engine. I measured and **rejected** the fancier ideas (weight tiering,
> MoE-tiering, prefill/decode disagg, P2P bypass) — they don't beat the standard engines on
> this hardware. The documented failures are in the repo (I think that's a feature, not a
> bug). What's real: the convenience, the heterogeneous-GPU orchestration, and the validated
> agent tool-calling.
>
> Repo + honest benchmarks: <lien>. Feedback welcome.

*(Joindre : le GIF asciinema de la session Aider + le tableau perf de BENCHMARK_RESULTS.md.)*

## Réponses aux objections prévisibles

**« Pourquoi pas le serveur llama.cpp directement ? »**
> VRAMancer L'UTILISE (llama.cpp est le backend GGUF). Il ajoute par-dessus : détection GPU
> + split hétérogène auto, sélection de backend auto (HF/llama.cpp/vLLM), auto-heal, parsing
> tool-call validé pour les agents, alias de modèle, dashboard/doctor/history. Pour un seul
> modèle GGUF sur un GPU, llama.cpp seul suffit ; l'intérêt est le multi-GPU dépareillé + l'UX.

**« Pourquoi pas Ollama ? »**
> Ollama est excellent pour du single-GPU/UX. Ce qu'il ne fait pas et que VRAMancer vise :
> orchestrer des GPU **dépareillés** (Ampere+Blackwell, tailles VRAM différentes) avec split
> proportionnel, et le cross-vendor/cross-nœud (en cours). Le function calling validé agent
> est comparable. Positionnement : « le multi-GPU hétérogène + l'agent local », pas « un
> meilleur Ollama ».

**« C'est juste un wrapper autour d'accelerate. »**
> Oui, en partie — et c'est assumé (README). accelerate fait le forward bf16. La valeur est
> la couche au-dessus (orchestration, anti-OOM, tool-calling agent, packaging) + les optims
> mesurées. Cf. Ollama = llama.cpp + UX, et ça marche.

## Checklist C7 (avant lancement)
- [ ] `VRM_CONTINUOUS_BATCHING=0` testé pour le coding (contexte plein) + re-mesure 8K/32K/64K.
- [ ] Repro session Aider **3×** de suite + capture asciinema.
- [ ] `python -m build` + `twine check` + install depuis TestPyPI sur venv vierge.
- [ ] Tag `v2.0.0` + GitHub Release avec la section CHANGELOG.
- [ ] `pip install vramancer` OK depuis PyPI.
- [ ] Post publié, premières questions traitées avec liens vers la doc.
