# C1 — Preuve d'usage Aider : findings (2026-07-05)

> Session Aider réelle contre le serveur local Qwen3.6-35B-A3B. Objectif Fable :
> « le contact avec un client réel révèle ce que les tests unitaires ne voient pas ».
> Résultat : **3 bugs bloquants trouvés + corrigés**, puis session complète RÉUSSIE.

## Setup
- `pip install aider-chat` (0.86.2), repo de test `benchmarks/agent_proof/`.
- `aider --model openai/qwen3.6-coder --openai-api-base http://localhost:5030/v1 --yes --no-git --no-stream --message "..."`.
- Tâche : « Add input validation to parse_config() and update the tests ».

## Bugs révélés (invisibles aux tests unitaires) et corrigés
1. **`max_tokens` défaut = 128** → l'édition de fichier était tronquée en pleine phrase.
   Fix : défaut 2048 (`VRM_DEFAULT_MAX_TOKENS`), cap 16384. *(commit max_tokens)*
2. **Nom de modèle = chemin GGUF complet** (`/home/.../model.gguf`) → moche/fragile pour
   Aider/LiteLLM. Fix : alias `VRM_MODEL_ALIAS` (défaut `qwen3.6-coder`) exposé dans
   `/v1/models` + réponses. *(commit alias)*
3. **L'alias déclenchait un reload** → `_ensure_model` rechargeait dès que `model` != nom
   réel ; Aider envoie l'alias → tentative de reload → 500 « vLLM not found ». Fix :
   ne recharger que si le modèle demandé != nom réel ET != alias. *(commit ensure_model)*

## Résultat final (après les 3 fixes)
- Aider a produit l'édition **whole format**, appliquée aux 2 fichiers.
- `config.py` : validation host (présence + non vide) + port (int valide → ValueError).
- `test_config.py` : 4 tests (missing/empty host, invalid port) — **4/4 passent**.
- Session end-to-end **sans intervention manuelle**.

## Acceptation Fable — état
- ✅ Session Aider aboutit (fichiers modifiés, tests mis à jour) sans intervention.
- ✅ **Reproductible 3×** (2026-07-05) : RUN 1/2/3 → 2 fichiers édités + tests passent (5/5)
  à chaque fois. Reset agent_proof entre chaque. Déterministe et correct.
- ⏳ Capture asciinema (GIF README) — action manuelle (Jérémie), pour le lancement.

## Observations pour la suite
- Le modèle a tendance à répondre en **français** + préambule verbeux avant l'édit (le
  format Aider « whole » l'a quand même parsé). À surveiller (pas bloquant ici).
- Format d'édition : Aider a utilisé « whole edit format » (réécriture complète du fichier),
  qui marche mieux qu'un diff avec un modèle quantifié. À noter dans la doc C6.
