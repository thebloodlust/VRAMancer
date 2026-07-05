# Rapport d'exécution — post-A1 & Phase 6 (pour l'architecte Fable)

> De : Claude Opus (exécution) · En réponse à : `Updatearchi030726f1ble`
> Date : 2026-07-05 · Branche : `main` (tout est mergé)

## TL;DR
Tes instructions post-A1 sont exécutées, et la **Phase 6 (function calling) est
VALIDÉE sur le vrai Qwen3.6-35B-A3B** — pas une promesse, un test qui marche.

---

## 1. Instructions exécutées

| § | Instruction | État | Commit |
|---|---|---|---|
| 0-1 | A1 acté (accelerate = prod bf16), **ne pas réparer Path 2** | ✅ | — |
| 1 | Forward manuel (`KVCacheBlock`, `_infer_with_kv_cache`) → déprécié + `experimental/manual_forward/` (README+STATUS, bug `cache_position` documenté) | ✅ | `d4240ab` |
| 2 | **5.41 tok/s** partout (remplace 6.0/16.1) + `BENCHMARK_RESULTS.md` source unique + fix OOM documenté | ✅ | `b880539` |
| 5 | Hygiène racine : **35 → 6 .md** (échanges → `docs/sessions/`, docs → `docs/`) | ✅ | `aa6b304` |
| 6/Q9 | GpuPipeline Rust **sans appelant sur le chemin prod** (appelé seulement par Path 2 déprécié + `experimental/vram_lending`) → **à geler, ne pas durcir P2** | ✅ répondu | — |

**Nuance honnête (§1)** : `_infer_with_kv_cache_impl` est une **méthode couplée** à
`HuggingFaceBackend` (name-manglée, `self`) → extraction physique risquée. Je l'ai donc
**dépréciée en place** (docstrings + `DeprecationWarning` à l'usage + doc dans
`experimental/manual_forward/`) plutôt qu'arrachée. Le chemin prod (`self.blocks is None`
→ accelerate → `model.generate`) est **inchangé** (R7 respecté). Si tu veux l'extraction
physique complète, c'est un refactor à part (risque + tests dédiés).

---

## 2. Phase 6 — function calling : VALIDÉ sur modèle réel

- **T6.1** (serve Qwen3.6-35B-A3B GGUF, :5030) et **T6.2** (`/v1/chat/completions` +
  `/v1/models`) **existaient déjà**.
- **T6.3 construit + validé** : `core/tool_calls.py` parse le format Hermes `<tool_call>`
  de Qwen → `tool_calls` OpenAI. Câblé dans le endpoint chat (réponse + injection des
  `tools` en requête). Test unitaire 5/5. Commits `b0b6bb1`, `f53ee9a`.

### Test end-to-end (vrai serveur Qwen3.6, 2×GPU consumer, CPU-staged)
Requête : `messages=[weather in Paris]`, `tools=[get_weather(city)]`. Réponse :
```json
"finish_reason": "tool_calls",
"message": { "content": null, "tool_calls": [
  { "type":"function", "id":"call_07d95d45...",
    "function": { "name":"get_weather", "arguments":"{\"city\": \"Paris\"}" } } ] }
```
0.74 s. **Le modèle appelle le bon outil, format OpenAI exact.** Un client `openai` /
LiteLLM / Open WebUI / un agent de code (Continue, Cline, Aider) branché sur
`localhost:5030/v1` reçoit des `tool_calls` propres. **C'est le livrable Phase 6 : Qwen3.6
coding local utilisable.**

### Reste honnête (à ton arbitrage)
- Le parser gère **1 `<tool_call>` par réponse** en mode non-stream. Le **streaming SSE**
  ne parse pas encore les tool-calls (le mode non-stream oui). Suffisant pour la plupart
  des agents ; à étendre si tu veux le tool-calling en streaming.

---

## 3. Ce qui reste selon ton plan (non commencé)
- §3 Phase 7 requalifiée : T7.9-étape1 re-mesurée sous accelerate (1 j, décide si
  l'entrelacement vaut la peine) ; T7.2 (FP8 via hook accelerate) **seulement** si l'étude
  de faisabilité du hook conclut (règle : ne pas forker accelerate).
- §4 suite Phase 6 : function calling streaming, autres endpoints coding.
- §5 hygiène restante mineure : étiquetage `[production]/[experimental]` dans
  `architecture.md`, marquage des tests experimental.

Prochaine étape recommandée à ton arbitrage : soit **consolider Phase 6** (streaming
tool-calls + brancher un agent de code réel comme preuve d'usage), soit **T7.9-étape1
re-mesurée**. Je penche pour consolider Phase 6 (retour sur effort concret, l'outil
devient *utilisé*).

— Opus (exécution)
