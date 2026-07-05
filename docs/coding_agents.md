# Agents de code locaux avec VRAMancer + Qwen3.6

Faire tourner un **agent de code** (Aider, Cline, Continue…) sur un modèle **local**,
sur des GPU consumer **dépareillés** (ex. RTX 3090 + RTX 5070 Ti), via une API
**OpenAI-compatible avec function calling**. Configs **validées** (session Aider réelle,
cf. `docs/sessions/C1_AIDER_FINDINGS.md`), pas théoriques.

## 1. Lancer le serveur
```bash
./serve_qwen36.sh          # Qwen3.6-35B-A3B GGUF sur 2 GPU, API sur :5030
```
Le serveur expose : `/v1/chat/completions`, `/v1/models`, `/health`. Le modèle est servi
sous l'alias **`qwen3.6-coder`** (`VRM_MODEL_ALIAS`).

## 2. Aider (validé end-to-end)
```bash
pip install aider-chat
export OPENAI_API_BASE=http://localhost:5030/v1
export OPENAI_API_KEY=dummy            # local, pas d'auth (voir sécurité plus bas)
aider --model openai/qwen3.6-coder --no-stream
```
Testé : édition multi-fichiers appliquée, tests générés passent, sans intervention.
`--no-stream` : le streaming des tool-calls n'est pas requis par Aider (mode non-stream OK).

## 3. Cline / Continue (VS Code)
Backend « OpenAI Compatible » :
- Base URL : `http://localhost:5030/v1`
- API key : n'importe quoi (`dummy`)
- Model : `qwen3.6-coder`

## 4. Client OpenAI brut (agent maison)
```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:5030/v1", api_key="dummy")
r = c.chat.completions.create(model="qwen3.6-coder",
    messages=[{"role":"user","content":"Weather in Paris? Use the tool."}],
    tools=[{"type":"function","function":{"name":"get_weather",
        "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}])
print(r.choices[0].message.tool_calls)   # -> get_weather({"city":"Paris"})
```

## Pièges rencontrés (déjà corrigés, documentés pour info)
- **max_tokens** : le défaut était 128 → éditions tronquées. Désormais **2048**
  (`VRM_DEFAULT_MAX_TOKENS`).
- **Nom de modèle** : utiliser l'alias `qwen3.6-coder` (pas le chemin GGUF).
- **Contexte** : ~4096 tokens/requête par défaut (continuous batching = 4 slots). Pour de
  gros fichiers, `VRM_CONTINUOUS_BATCHING=0` (tout le contexte à une requête) et/ou un
  `n_ctx` plus grand. Un prompt trop long renvoie une **erreur 400 claire**, pas un crash.
- **Blocs de raisonnement** : Qwen3.6 émet des `<think>` ; VRAMancer les retire du
  `content` (exposés en `reasoning_content`) et désactive le thinking pour le tool-calling.

## Performances (mesurées sur CE chemin — llama.cpp/GGUF)
| Prompt (~tok) | TTFT | décode tok/s |
|---|---|---|
| 1 000 | 0.31 s | ~305 |
| 4 000 | 0.77 s | ~246 |

> Honnêteté : ces chiffres sont pour le chemin **llama.cpp/GGUF** (Qwen3.6). L'optimisation
> **prompt-lookup +500%** de VRAMancer a été mesurée sur le backend **HuggingFace**, PAS
> sur ce chemin llama.cpp — ne pas l'attribuer ici. Cf. `BENCHMARK_RESULTS.md`.

## Sécurité
- Le serveur écoute par défaut sur toutes les interfaces via `serve_qwen36.sh` (usage LAN
  perso). Pour du strictement local : `./serve_qwen36.sh --host 127.0.0.1`.
- Exposition réseau : définir `VRM_API_TOKEN` (auth Bearer) et passer derrière un reverse
  proxy TLS. Ne jamais exposer sans token.
