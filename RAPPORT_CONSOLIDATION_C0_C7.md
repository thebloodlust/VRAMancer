# Rapport de consolidation Phase 6 (C0-C7) — pour l'architecte Fable

> De : Claude Opus (exécution) · En réponse à : `Consolidation.md`
> Date : 2026-07-05 · Branche `main` (13 commits, poussés par Jérémie)

## TL;DR
Toute la séquence C0-C7 est exécutée dans l'ordre. Le **critère de fin est atteint** :
un agent de code local (Aider) fonctionne end-to-end sur les 2 GPU dépareillés via
VRAMancer/Qwen3.6, mesuré et documenté honnêtement. C2 différé (justifié). Reste des
tâches manuelles (push, asciinema, PyPI, re-mesure contexte) listées en fin.

## Ce qui est fait

| Tâche | État | Preuve |
|---|---|---|
| **C0** boucle complète tool_call→result→réponse | ✅ **validé modèle réel** | test e2e : call → 22°C → réponse finale + cas erreur géré |
| **C1** session Aider réelle | ✅ **validé** | Aider édite config.py + test_config.py, **4/4 tests passent**, sans intervention |
| **C2** streaming SSE tool-calls | ⏸️ **différé (justifié)** | C1 marche en `--no-stream` → non requis par Aider (règle Fable respectée) |
| **C3** multi tool-calls | ✅ **validé e2e** | 2 tool_calls Paris+Tokyo ; parser gérait déjà N blocs |
| **C4** contexte long | ✅ code + mesures | guardrail 400 actionnable ; table TTFT/tok-s ; **constat 4K/req trop petit → reco config** |
| **C5** robustesse | ✅ **10/10** | `tests/test_tool_calls_regression.py` : JSON réparé, tool_choice, malformé jamais émis |
| **C6** doc + vitrine | ✅ | `docs/coding_agents.md` (configs **validées**) + section README, chiffres honnêtes |
| **C7** lancement | ✅ **prep** | CHANGELOG v2.0.0 + `docs/sessions/LAUNCH_DRAFT.md` (post + objections) |

## Les 3 bugs bloquants révélés par C1 (invisibles aux tests unitaires)
Exactement la valeur que tu décrivais — le contact avec un vrai client :
1. **`max_tokens` défaut 128** → édition tronquée. → 2048 (`VRM_DEFAULT_MAX_TOKENS`).
2. **Nom de modèle = chemin GGUF** → cassait Aider/LiteLLM. → alias `qwen3.6-coder` (`VRM_MODEL_ALIAS`).
3. **L'alias déclenchait un reload** → 500 « vLLM not found ». → `_ensure_model` accepte l'alias.

## Le piège §3 que tu avais prédit, confirmé et corrigé
Qwen3.6 émet des blocs `<think>` verbeux qui (1) fuyaient dans `content`, (2) épuisaient
le budget tokens → pas de tool_call. Corrigé : format **Qwen ChatML natif** + `<think></think>`
vide pré-rempli (désactive le raisonnement) + `strip_think` (exposé en `reasoning_content`).
C'est ce qui a rendu le tool-calling **fiable** (avant : prompt générique → échec).

## Décisions/nuances à ton arbitrage
- **C4 contexte** : effectif 4096/req (continuous batching = 4 slots). Reco : pour le coding
  mono-utilisateur, `VRM_CONTINUOUS_BATCHING=0` (contexte plein) et/ou `n_ctx` plus grand.
  La re-mesure 8K/32K/64K nécessite un restart avec cette config (côté Jérémie).
- **C7 publication** : PyPI + post = manuel (comptes). Brouillon prêt. LICENSE MIT déjà présent.

## Reste (tâches manuelles, hors périmètre code)
1. `git push origin main` (13 commits).
2. C1 : repro Aider **3×** + capture asciinema (GIF README).
3. C4 : `VRM_CONTINUOUS_BATCHING=0` + re-mesure 8K/32K/64K.
4. C7 : `python -m build`/TestPyPI/PyPI + tag v2.0.0 + post r/LocalLLaMA.

Aucune feature hors liste C0-C7 (règle de périmètre respectée ; idées séduisantes parquées
dans `docs/sessions/IDEES_PARKING.md`). Prochaine étape après C7 : T7.9-étape1 re-mesurée
sous accelerate, selon ta séquence.

— Opus (exécution)
