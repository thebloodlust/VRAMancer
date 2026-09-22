# Plan D — Post-lancement (conçu pour exécution par IA limitée)

> **Pour l'agent exécutant :** ce plan est écrit pour être suivi À LA LETTRE, sans
> interprétation. Chaque tâche est atomique : une tâche = un commit. Si une validation
> échoue → ARRÊTE-TOI et signale à l'utilisateur. Ne brute-force jamais.
>
> **Auteur :** Audit Fable 2026-08-04 · **Base :** `main` @ `4187e2f` (v2.0.0, post-C7)

---

## Règles ABSOLUES (répétées de plans précédents, toujours valides)

1. **Répertoire de travail** : `/home/jeremie/VRAMancer/VRAMancer` (le checkout IMBRIQUÉ).
   Le parent `/home/jeremie/VRAMancer/` contient un vieux checkout périmé — NE JAMAIS y toucher.
2. **NE JAMAIS** modifier : `_deprecated/`, `tests/test_chaos_concurrency.py`,
   `csrc/paged_attention_kernel.cu`, `core/security/__init__.py`, `core/paged_attention.py`.
3. **NE JAMAIS** push, merge, tagger, ni publier quoi que ce soit — c'est Jérémie qui pousse.
4. **NE JAMAIS** désactiver un test pour faire passer la suite.
5. **Honnêteté chiffrée** : tout claim de perf dans un .md doit venir d'une MESURE faite
   dans la session, copiée-collée. Pas d'extrapolation, pas de "devrait donner".
6. **Suite de tests de référence** (à lancer avant ET après chaque tâche qui touche au code) :
   ```bash
   cd /home/jeremie/VRAMancer/VRAMancer && source .venv/bin/activate
   VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 VRM_BACKEND_ALLOW_STUB=1 \
     pytest tests/ --ignore=tests/test_chaos_concurrency.py \
     --timeout=60 --timeout-method=signal --tb=no --no-cov -q 2>&1 | tail -3
   ```
   Échecs pré-existants TOLÉRÉS (ne pas tenter de les fixer) : 2× nvfp4, chaos_concurrency.
   Tout AUTRE échec nouveau = ta modification a cassé quelque chose → revert et signale.
7. Si tu touches à `rust_core/` : `maturin develop` ne remplace PAS le `.so` importé
   (paquet shadow). Vérifie le comportement réel, pas juste la compilation.
8. Préfixe de commit : `[D<x>.<y>]`. Fin de message :
   `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` (ou l'agent exécutant).

---

## État d'avancement au 2026-09-22 (session Opus 5)

| Tâche | État | Où |
|---|---|---|
| D0.1 token dans le remote | ✅ côté agent (remote nettoyé) — **révocation GitHub à faire par Jérémie** | `.git/config` |
| D0.2 mojibake du post | ✅ | `docs/sessions/POST_LANCEMENT_FINAL.md` |
| D0.3 conflit venv Aider | ⏸ inchangé (à faire hors session de bench) | — |
| D1 checklist lancement | ⏸ MANUEL (PyPI, tag, post) | — |
| D2.1 quickstart tool calling | ✅ | `examples/tool_calling_quickstart.py` |
| D2.2 métriques tool-calls | ✅ 3 compteurs + 5 tests | `core/metrics.py`, `core/tool_calls.py` |
| D2.3 dashboard GPU réel | ✅ **le constat du plan était faux** : pas de GPU factice dans les templates ; le vrai défaut était que les cartes AMD étaient invisibles | `dashboard/dashboard_web.py` |
| D2.4 POC WebGPU | ✅ **sans objet** : déjà dans `_deprecated/`, plus aucun import depuis `core/` | — |
| D2.5 docstrings orchestrator | ✅ 3 modules, zéro ligne de code | `core/orchestrator/` |
| D2.6 doctor --share | ✅ | `core/doctor.py` |
| D2.7 README failure-reports + compat | ✅ + template d'issue | `README.md`, `.github/ISSUE_TEMPLATE/` |
| D2.9 tests sécurité | ✅ 13 tests (verify_request avait déjà 10 tests ; c'est `enforce_startup_checks` qui n'en avait aucun) | `tests/test_security_startup_checks.py` |
| D3.b 2e machine | ⏸ pas de matériel | — |
| D3.c simulation Docker | ✅ mesuré (UDP **et** mDNS traversent le bridge) | `tests/integration/docker_cluster_sim.sh` |
| D3.d paire mixte | ✅ **MESURÉ et TRANCHÉ** : cross-vendor maison = non-sujet (paire 32% plus lente quand le modèle tient ; +28% mais inutilisable quand il ne tient pas) | `docs/reports/D3D_CROSS_VENDOR_VERDICT.md` |
| D3.a ROCm / bridge python | ⛔ **à ne PAS entreprendre** — D3.d a répondu : rien à gagner | idem |

Hors plan, trouvé en faisant tourner le code sur la 7900 XT (3 pannes réelles) :
téléchargement automatique de llama-server cassé pour tout le monde (404),
extraction produisant un binaire qui ne démarre pas, options llama.cpp changées
de forme — plus les GPU AMD invisibles de tout l'outillage. Voir `conclusion7900xt.md`
à la racine et les commits `[AMD]`.

---

## D0 — Hygiène & sécurité (À FAIRE EN PREMIER)

### D0.1 — Token GitHub en clair dans le remote *(S, MANUEL Jérémie + agent)*
`git remote -v` montre un token `ghp_…` embarqué dans l'URL. Il est stocké en clair
dans `.git/config` et fuit dans tout log/copie d'écran.
- **Jérémie (manuel)** : révoquer ce token sur github.com → Settings → Developer settings
  → Personal access tokens, en créer un nouveau.
- **Agent** :
  ```bash
  git remote set-url origin https://github.com/thebloodlust/VRAMancer.git
  git config credential.helper store   # le nouveau token sera saisi une fois au 1er push
  ```
- **Validation** : `git remote -v` ne montre plus AUCUN token. Pas de commit (config locale).

### D0.2 — Fichier `Post lancement` : nom + encodage *(S)*
Le fichier `Post lancement` (racine, espace dans le nom) est en mojibake
(UTF-8 double-encodé : « â€” » au lieu de « — »).
- Réencoder : `iconv -f utf-8 -t latin1 "Post lancement" | iconv -f utf-8 -t utf-8 > docs/sessions/POST_LANCEMENT_FINAL.md` — puis OUVRIR le fichier et vérifier visuellement que les tirets/accents sont corrects. Si le résultat est pire, faire la correction à la main (remplacer â€” par —, Ã© par é, etc.).
- `git rm "Post lancement"`.
- **Validation** : `grep -c "â€" docs/sessions/POST_LANCEMENT_FINAL.md` retourne 0.
- Commit : `[D0.2] déplace/réencode le post de lancement (mojibake + espace dans le nom)`.

### D0.3 — Conflit de venv : Aider épingle huggingface-hub==1.4.1 *(S, note)*
`transformers` exige `huggingface-hub>=1.5.0` ; `aider-chat` (installé dans le MÊME venv)
épingle `==1.4.1`. Corrigé le 2026-08-04 en installant hub 1.26.0 (test_scheduler repasse,
`aider --version` marche encore), mais le conflit pip persiste. Solution propre quand
l'occasion se présente : installer Aider via `pipx install aider-chat` (venv séparé) et le
retirer du venv projet. Ne PAS le faire en pleine session de bench.

---

## D1 — Checklist lancement (MANUEL Jérémie, l'agent ne peut PAS le faire)

État vérifié 2026-08-04 : commits poussés ✅ · GIF ✅ · triple repro Aider ✅ ·
re-mesure 4K-16K ✅ · **PyPI ✗ (paquet absent)** · **tag v2.0.0 ✗** · **post non publié ✗**.

1. Re-mesure 32K/64K si le post les cite (sinon retirer ces lignes du tableau — règle 5).
2. Remplir les `[À REMPLIR]` de `docs/sessions/POST_LANCEMENT_FINAL.md`.
3. `python -m build` → TestPyPI → PyPI ; puis sur un venv VIERGE, exécuter LITTÉRALEMENT
   les 4 lignes du bloc Setup du post.
4. `git tag v2.0.0 && git push --tags` + GitHub Release.
5. Publier (14h-16h Paris) + poster le commentaire "rapport A1" dans la foulée.

---

## D2 — Tâches mécaniques pour IA limitée (indépendantes, dans l'ordre de préférence)

### D2.1 — `examples/tool_calling_quickstart.py` *(S)*
Idée parkée validée par l'architecte (coût marginal). Écrire ~30 lignes : client
`openai` officiel pointé sur `http://localhost:5030/v1`, une fonction-outil
`get_weather(city)`, boucle 2 tours (tool_call → tool_result → réponse finale).
- Modèle : s'inspirer de `tests/test_tool_calls_regression.py` pour le format.
- Le script doit marcher SANS serveur lancé en affichant une erreur claire
  (« Lance d'abord: vramancer serve … ») — try/except sur la connexion.
- **Validation** : `python examples/tool_calling_quickstart.py` sans serveur → message
  clair, exit code 1. Pas de crash traceback brut.
- Commit : `[D2.1] examples/tool_calling_quickstart.py (porte d'entrée dev)`.

### D2.2 — Métriques agent dans `/metrics` *(S/M)*
Idée parkée « coût faible ». Dans `core/metrics.py`, ajouter 3 compteurs Prometheus :
`VRM_TOOL_CALLS_EMITTED`, `VRM_TOOL_CALLS_MALFORMED` (réparés par le parser),
`VRM_TOOL_CALLS_FAILED`. Les incrémenter dans le parser de tool-calls
(chercher `tool_call` dans `core/api/` — suivre le chemin du code C5).
- **Validation** : la suite de référence passe + un test unitaire nouveau qui vérifie
  que le compteur s'incrémente quand le parser répare un JSON malformé
  (réutiliser les fixtures de `tests/test_tool_calls_regression.py`).
- Commit : `[D2.2] métriques agent tool-calls dans /metrics`.

### D2.3 — Dashboard : retirer le GPU data hardcodé *(M)*
`dashboard/` templates contiennent des données GPU factices. Les remplacer par des
appels à `/api/gpu` réel. NE PAS redesigner — remplacement minimal des valeurs
hardcodées par le fetch.
- **Validation** : lancer le dashboard en mode stub (`VRM_TEST_MODE=1`), vérifier
  qu'aucune valeur factice codée en dur ne s'affiche (grep des anciennes valeurs
  dans les templates → 0 occurrence).
- Commit : `[D2.3] dashboard branché sur /api/gpu réel (fin du GPU data factice)`.

### D2.4 — Supprimer le POC WebGPU mort *(S)*
`core/backends_webgpu.py` est marqué « Production Ready = FAUX » dans l'audit et
n'a pas de consommateur réel. Le déplacer vers `_deprecated/` N'EST PAS autorisé
(règle 2) — le déplacer vers `experimental/` à la place, et mettre à jour tout
import qui le référence.
- **Validation** : `grep -rn "backends_webgpu" core/ vramancer/ tests/` → plus aucune
  référence cassée ; suite de référence passe.
- Commit : `[D2.4] backends_webgpu POC déplacé vers experimental/`.

### D2.5 — Docstrings honnêtes, modules grade C *(M, mécanique)*
Continuer le travail commencé (cross_vendor, RemoteExecutor). Pour CHAQUE fichier de
`core/orchestrator/` : lire le docstring de tête, le comparer à ce que le code fait
RÉELLEMENT, corriger toute sur-promesse (« production-ready », « zero-copy »,
« automatic » non câblé). Ne toucher QUE les docstrings/commentaires, zéro code.
- **Validation** : `git diff --stat` ne montre que des .py avec modifs de docstrings ;
  suite de référence passe (elle ne doit même pas bouger).
- Commit : `[D2.5] docstrings honnêtes core/orchestrator/`.

### D2.6 — `vramancer doctor --share` *(S)*
Ajouter un flag `--share` à la commande doctor existante : dump diagnostic ANONYMISÉ
(GPUs + VRAM, driver, version CUDA, OS, version vramancer, backend détecté) au format
markdown collable dans une issue GitHub. AUCUNE donnée perso (pas de hostname, pas de
chemins, pas d'IP).
- **Validation** : sortie inspectée à l'œil → zéro info identifiante ; suite de référence passe.
- Commit : `[D2.6] vramancer doctor --share (dump diagnostic anonymisé pour issues)`.

### D2.7 — README : badge « failure reports » + table compat community-tested *(S)*
1. Dans le README, section visible en haut : lien direct vers les 6 rapports négatifs
   (tiering, MoE, disagg, P2P…) présenté comme argument de crédibilité.
2. Table « Clients agents » : Aider = ✅ validé e2e ; Cline / Continue / OpenWebUI =
   « non testé — PR bienvenue » avec lien vers un template d'issue de compat.
- **Validation** : liens cliqués, zéro lien mort.
- Commit : `[D2.7] README badge failure-reports + table compat community-tested`.

### D2.9 — Tests sécurité manquants *(M)*
TODO §6 : écrire des tests pour `verify_request()` et `enforce_startup_checks()`
(chercher dans `core/security/`). Cas minimum : requête sans token → 401 ;
token invalide → 401 ; token valide → 200 ; startup check en mode prod sans
mot de passe admin → refus.
- **Validation** : nouveaux tests passent + suite de référence inchangée.
- Commit : `[D2.9] tests sécurité verify_request + startup_checks`.

---

## D3 — En attente de MATÉRIEL (ne PAS commencer sans)

### D3.a — GPU AMD (cross-vendor)
Préparé, jamais exécuté sur vrai hardware. Le jour où le GPU AMD arrive :
1. ROCm installé + `python -c "import torch; print(torch.version.hip)"` non-null.
2. `pytest tests/test_cross_vendor.py -v` en réel (pas en mock).
3. Mesurer le débit réel de Strategy 2 (Pipelined) — seule stratégie annoncée fiable.
   Comparer aux 25-50 GB/s du docstring ; corriger le docstring si l'écart est grand.
4. Statuts Strategy 0 (DMA-BUF) et 1 (ReBAR) restent « partial » tant que non finis.
5. AUCUN chiffre de `docs/sessions/TOK_S_ET_AMD_XGMI.md` n'est mesuré — ne jamais
   les citer comme résultats.

### D3.b — 2e machine (cross-nœud)
1. Découverte mDNS puis fallback UDP entre les 2 machines réelles.
2. Heartbeat / join / leave en conditions réelles.
3. Mesure bande passante inter-nœud réelle (LAN/Thunderbolt selon setup).
4. Seulement ENSUITE envisager le sharding cross-node (`BlockOrchestrator`, XL).

### D3.c — Préparation SANS matériel : cross-nœud simulé en Docker *(M, faisable maintenant)*
Deux conteneurs sur la même machine, réseau bridgé, pour dégrossir `cluster_discovery.py`
avant l'arrivée de la 2e machine : découverte UDP broadcast (mDNS peut ne pas traverser
le bridge Docker — si c'est le cas, le NOTER, c'est un résultat utile), heartbeat,
join/leave en tuant un conteneur.
- **Validation** : un script `tests/integration/docker_cluster_sim.sh` reproductible +
  résultat consigné dans un .md (ce qui marche / ce qui ne marche pas en bridge).
- Ceci ne REMPLACE pas D3.b : le noter explicitement dans le .md.
- Commit : `[D3.c] simulation cross-nœud 2 conteneurs Docker (dégrossissage avant 2e machine)`.

> **État 2026-09-18 (7900 XT installée)** : les 2 cartes sont vues sur le bus PCI de la VM,
> MAIS (1) le kernel est passé à 6.8.0-137 au reboot et le module nvidia (595-open) n'existe
> que pour -134 → nvidia-smi mort ; (2) la 7900 XT s'est retrouvée `runtime_status=error`
> après un échec de resume runtime-PM (`resume of IP block <smu> failed -62`, bug Navi 3x
> en passthrough) → device inutilisable jusqu'au reboot ; fix durable : `amdgpu.runpm=0`
> en cmdline kernel. Procédure de réparation donnée à Jérémie (1 session sudo + 1 reboot).
> Le bench D3.d est PRÊT : `benchmarks/bench_vulkan_pair.sh` (build Vulkan précompilé
> b11026 dans `~/tools/llama-vulkan-b11026`, RADV déjà installé — ROCm PAS nécessaire
> pour D3.d). ROCm/torch-hip nécessaires seulement pour D3.a (tests bridge python).

### D3.d — AVANT tout investissement cross-vendor : mesurer llama.cpp Vulkan *(S, dès GPU AMD)*
llama.cpp a un backend Vulkan capable de mélanger les vendors, et un mode RPC multi-machine.
Le jour du GPU AMD, AVANT de câbler quoi que ce soit dans `cross_vendor_bridge.py` :
1 journée max pour mesurer ce que llama.cpp Vulkan donne déjà sur la paire NVIDIA+AMD
(mêmes modèle/quant/prompts que BENCHMARK_RESULTS.md). Si ça marche bien → le cross-vendor
maison est un non-sujet (scénario A1 : ne pas re-payer cette leçon). Si c'est cassé/lent →
la niche existe, documenter les chiffres et SEULEMENT ALORS investir.

---

## D4 — Pistes stratégiques (décision Jérémie UNIQUEMENT — pas pour IA limitée)

> Issues de l'audit 2026-08-04. Une IA limitée ne doit PAS commencer ces chantiers ;
> ils changent le périmètre du projet.

1. **Extraire le proxy « fiabilité agent » en projet autonome** *(recommandation n°1)*.
   Le travail C0-C5 (normalisation des formats de tool-calls, réparation JSON, strip des
   think-blocks, défauts sains type max_tokens) extrait en un proxy mince (~2 000 lignes)
   qui se place devant N'IMPORTE QUEL backend OpenAI-compatible (llama.cpp, Ollama, vLLM,
   cloud). Petit, unique, réutilise le seul avantage démontré du projet, surfe la vague
   agents au lieu de concurrencer Ollama.
2. **Base crowdsourcée « quel setup fait quel tok/s »**. Harness de bench standardisé +
   agrégation des soumissions communautaires (GPU dépareillés × modèle × quant → tok/s).
   Effet de réseau, peu de code, prolonge la marque « benchmarks honnêtes ».
   D2.6 (`doctor --share`) en est la brique de départ.
3. ~~**Cross-vendor maison**~~ : **FERMÉ le 2026-09-22.** D3.d a été mesuré sur la
   vraie paire 3090 + 7900 XT : llama.cpp Vulkan mélange déjà les deux vendeurs, et il
   n'existe aucun régime où un pont maison apporterait quelque chose (voir
   `D3D_CROSS_VENDOR_VERDICT.md`). Ne pas rouvrir sans un cas nouveau : un modèle qui
   tiendrait ENTIÈREMENT dans la VRAM cumulée de deux cartes de marques différentes.
4. **NE PAS** : réécrire un moteur d'inférence (terrain le plus contesté, réfuté par A1) ;
   swarm/ledger (XL, marché encombré — gel jusqu'à demande d'early adopters).

---

## Ce qu'on ne fait PAS (gel confirmé)

- Streaming SSE des tool-calls (réveil : un client cible l'exige — vérifier 1 h avant de coder).
- Multi-modèles simultanés, routage local/cloud, RAG intégré, UI web de chat (gel architecte §3).
- Toute réécriture Rust de ce qui marche en Python (le gain est de câbler l'existant,
  pas de réécrire).
- Tout claim tiering/MoE/disagg/P2P : réfutés par mesure (6 rapports négatifs) — ne pas rouvrir.
