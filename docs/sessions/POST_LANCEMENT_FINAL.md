# Post de lancement r/LocalLLaMA — version finale corrigée (architecte)

> Corrections intégrées vs brouillon Opus : (1) suppression du "(en cours)" cross-vendor,
> (2) commande vitrine alignée sur le package pip, (3) emplacements balisés pour les
> chiffres réels [À REMPLIR après re-mesure], (4) lien repo.
> À publier par Jérémie après la checklist de fin de document.

---

## Titre

Local coding agent on mismatched consumer GPUs (RTX 3090 + RTX 5070 Ti): Qwen3.6-35B-A3B with OpenAI-compatible tool calling, one command

## Corps

I've been building **VRAMancer** — an orchestrator that runs big models across
mismatched consumer GPUs (different generations, different VRAM sizes, no NVLink)
and exposes an OpenAI-compatible API with working function calling, so tools like
Aider just work against a local model.

On a 3090 + 5070 Ti (Ampere + Blackwell, P2P blocked by the platform), it serves
Qwen3.6-35B-A3B (GGUF, [À REMPLIR: quant utilisé, ex. Q4_K_M]) and I validated a
full Aider session end-to-end: it edits code and updates the tests, unassisted.
The tool-call round-trip (call → tool result → final answer) is handled server-side,
including malformed-JSON repair and think-block stripping (Qwen3.6's reasoning blocks
were leaking into content and eating the token budget — fixed).

**Numbers on this pair** (coding profile, batching off, full context):

| Context | TTFT | Decode |
|---|---|---|
| 8K  | [À REMPLIR] s | [À REMPLIR] tok/s |
| 32K | [À REMPLIR] s | [À REMPLIR] tok/s |
| 64K | [À REMPLIR] s | [À REMPLIR] tok/s |

VRAM used: [À REMPLIR] GB / 40 GB total.

Setup:

```
pip install vramancer
vramancer serve --profile coding
# then point your agent at http://localhost:5030/v1
aider --openai-api-base http://localhost:5030/v1 --openai-api-key dummy --model openai/qwen3.6-coder
```

**Honest about what it is**: it's an orchestration + UX layer on top of
accelerate/llama.cpp, not a new inference engine. I measured and rejected the
fancier ideas (custom pipeline forward, weight tiering, temporal delta encoding,
P2P tricks) — they don't beat the standard engines on this hardware, and the
documented failure reports are in the repo. I think that's a feature, not a bug.
What's real: the heterogeneous-GPU orchestration, the auto-heal (OOM recovery
ladder — the server never dies), a measured +500% prompt-lookup speedup on the
HF backend for repetitive code generation, and the agent-validated tool calling.

Repo + honest benchmarks (including the failures): https://github.com/thebloodlust/VRAMancer

Feedback welcome — especially from anyone else running mismatched pairs.

*(Joindre : le GIF asciinema de la session Aider + capture du tableau perf.)*

---

## Premier commentaire à poster soi-même (aimant à crédibilité)

For anyone curious about the "documented failures" part: here's the parity report
where I benchmarked my own custom multi-GPU pipeline against accelerate, found it
was broken AND wouldn't win even if fixed, and killed it:
[lien vers le rapport A1 dans le repo]. Building it taught me more than shipping
it would have.

---

## Réponses prêtes aux objections

**« Pourquoi pas le serveur llama.cpp directement ? »**
VRAMancer l'utilise (llama.cpp est le backend GGUF). Il ajoute par-dessus :
détection GPU + split hétérogène auto, sélection de backend auto (HF/llama.cpp/vLLM),
auto-heal OOM, parsing tool-call validé sur un agent réel, alias de modèle,
profils de serve. Pour un seul modèle GGUF sur un GPU, llama.cpp seul suffit —
l'intérêt est le multi-GPU dépareillé + l'UX agent.

**« Pourquoi pas Ollama ? »**
Ollama est excellent en single-GPU/UX. Ce que VRAMancer vise et qu'Ollama ne
couvre pas : orchestrer des GPUs dépareillés (Ampere + Blackwell, tailles VRAM
différentes) avec split proportionnel et récupération d'OOM. Positionnement :
« le multi-GPU hétérogène + l'agent local », pas « un meilleur Ollama ».
[SUPPRIMÉ vs brouillon : le "cross-vendor/cross-nœud (en cours)" — gelé en
experimental/, non testé, ne JAMAIS le citer en argument tant que non prouvé.]

**« C'est juste un wrapper autour d'accelerate. »**
Oui, en partie — et c'est assumé dans le README. accelerate fait le forward bf16.
La valeur est la couche au-dessus (orchestration, anti-OOM, tool-calling agent,
packaging) + les optimisations mesurées. Ollama = llama.cpp + UX, et ça marche.

**« What tok/s? »** (arrivera même si les chiffres sont dans le post)
Renvoyer au tableau + lien BENCHMARK_RESULTS.md, préciser le quant et le profil.

**« Does it work with Cline / Continue / OpenWebUI? »**
Réponse honnête : validé end-to-end avec Aider (non-stream). Les autres clients
devraient fonctionner via l'API OpenAI mais n'ont pas été validés un par un ;
le streaming des tool-calls n'est pas encore implémenté (différé jusqu'à ce qu'un
client le requière). Invitation à remonter les incompatibilités en issue.

---

## Checklist finale avant de cliquer "Post"

- [ ] git push origin main (les 13 commits) — rien n'existe tant que ce n'est pas poussé
- [ ] Re-mesure 8K/32K/64K avec profil coding → remplir les [À REMPLIR] du post
- [ ] `pip install vramancer` depuis PyPI sur venv vierge → puis exécuter LITTÉRALEMENT
      les 4 lignes du bloc Setup du post, telles quelles — elles doivent marcher
- [ ] Vérifier que l'alias `--profile coding` existe bien sous ce nom (sinon aligner
      le post sur la vraie syntaxe)
- [ ] Cliquer tous les liens du README depuis un navigateur privé (le nettoyage a
      déplacé des fichiers ; zéro lien mort toléré)
- [ ] Tag v2.0.0 + GitHub Release publiée
- [ ] GIF uploadé et visible dans le README
- [ ] Publier un matin US (14h-16h heure de Paris), rester disponible 2 h pour
      répondre vite aux premiers commentaires
- [ ] Poster le premier commentaire (rapport A1) dans les minutes qui suivent
