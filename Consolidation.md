# Consolidation Phase 6 â€” Instructions de l'architecte

> De : l'architecte Â« Fable Â» Â· Ã€ : Claude Opus (exÃ©cution)
> En rÃ©ponse Ã  : `RAPPORT_EXECUTION_FABLE_A1_PHASE6.md` (2026-07-05)
> Objet : arbitrage rendu (consolider la Phase 6), tÃ¢ches ordonnÃ©es, idÃ©es
> additionnelles, prÃ©paration du lancement. T7.9-Ã©tape1 est REPORTÃ‰ aprÃ¨s C6.

---

## 0. Validation de ton rapport

- DÃ©prÃ©ciation EN PLACE du forward manuel plutÃ´t qu'extraction physique : **APPROUVÃ‰**.
  Ton raisonnement (mÃ©thode name-manglÃ©e couplÃ©e Ã  self â†’ refactor risquÃ©) est correct ;
  l'objectif rÃ©el (que personne n'emprunte la voie cassÃ©e) est atteint sans risque.
  L'extraction physique n'aura lieu que si un besoin concret l'exige un jour. Bonne
  dÃ©sobÃ©issance intelligente â€” c'est exactement le jugement attendu.
- Q9 (GpuPipeline Rust sans appelant prod â†’ hardening P2 gelÃ©) : **ACTÃ‰**.
- Phase 6 T6.3 validÃ©e sur modÃ¨le rÃ©el en 0.74 s : c'est le livrable le plus important
  de l'histoire du projet. La suite ci-dessous le transforme en outil UTILISÃ‰.

## 1. ARBITRAGE : consolider la Phase 6. T7.9/T7.2 attendent.

Justification : optimiser un moteur que personne n'utilise encore rapporte moins que
rendre l'outil adoptable. T7.9-Ã©tape1 (re-mesure de bulle sous accelerate) ne dÃ©marre
qu'aprÃ¨s C6, sauf instruction contraire.

**RÃ¨gle de pÃ©rimÃ¨tre pour toute cette consolidation** : AUCUNE nouvelle feature hors
de la liste C0-C7. Si une idÃ©e sÃ©duisante apparaÃ®t en route, elle va dans
`docs/sessions/IDEES_PARKING.md`, pas dans le code.

---

## 2. TÃ¢ches de consolidation, dans l'ordre

### C0 â€” Boucle complÃ¨te tool-call â†’ tool_result â†’ rÃ©ponse finale (PRIORITÃ‰ 1, ~1 jour)
Ton test e2e valide la PREMIÃˆRE moitiÃ© de la boucle (le modÃ¨le Ã©met un tool_call).
Un agent rÃ©el fait l'aller-retour COMPLET : il exÃ©cute l'outil et renvoie un message
`role:"tool"` avec le rÃ©sultat ; le modÃ¨le doit alors produire la rÃ©ponse finale.
1. VÃ©rifier que le endpoint accepte les messages `{"role":"tool","tool_call_id":...,
   "content":...}` et que le chat template Qwen3.6 les injecte correctement
   (format Hermes : bloc <tool_response> selon le template du modÃ¨le â€” vÃ©rifier le
   template GGUF embarquÃ© avec llama.cpp, ne pas supposer).
2. Test e2e aller-retour : weather â†’ tool_call â†’ renvoyer {"temp": "22Â°C"} â†’
   le modÃ¨le rÃ©pond en langage naturel en utilisant 22Â°C.
3. Cas d'erreur : tool_result contenant une erreur ("city not found") â†’ le modÃ¨le
   doit gÃ©rer gracieusement, pas boucler sur le mÃªme tool_call Ã  l'infini
   (si boucle dÃ©tectÃ©e : max 3 appels identiques consÃ©cutifs â†’ forcer une rÃ©ponse texte).
**Acceptation** : test automatisÃ© de la boucle complÃ¨te (2 tours) qui passe 5/5,
incluant le cas d'erreur. Sans C0, AUCUN agent rÃ©el ne fonctionnera â€” c'est le
chaÃ®non manquant invisible dans le test actuel.

### C1 â€” Preuve d'usage rÃ©elle : une session Aider complÃ¨te (PRIORITÃ‰ 1)
1. Installer Aider, le configurer sur le serveur local :
   `aider --openai-api-base http://localhost:5030/v1 --openai-api-key dummy --model openai/<nom-servi>`
   (le nom du modÃ¨le doit correspondre Ã  /v1/models ; ajouter un alias si besoin, cf. C5).
2. TÃ¢che de rÃ©fÃ©rence reproductible : dans un petit repo de test (crÃ©er
   `benchmarks/agent_proof/` avec 3 fichiers Python simples), demander Ã  Aider :
   "Add input validation to parse_config() and update the tests accordingly" â€”
   une Ã©dition multi-fichiers avec aller-retour d'outils.
3. Documenter TOUT ce qui casse (format, timeout, contexte, stop tokens) et corriger
   au fil de l'eau. C'est l'objectif : le contact avec un client rÃ©el rÃ©vÃ¨le ce que
   les tests unitaires ne voient pas.
4. Si Aider ne passe pas en non-stream, basculer sur C2 (streaming) puis revenir.
**Acceptation** : la session Aider aboutit de bout en bout (fichiers modifiÃ©s,
tests mis Ã  jour) SANS intervention manuelle, reproductible 3 fois de suite.
Capturer la session en asciinema (`asciinema rec`) â†’ c'est le futur GIF du README.

### C2 â€” Tool-calls en streaming SSE (si C1 le rÃ©vÃ¨le nÃ©cessaire, sinon aprÃ¨s C3)
1. D'abord VÃ‰RIFIER si les clients cibles (Aider, Cline, Continue) exigent le
   streaming pour les tool-calls ou acceptent le non-stream. Ne pas implÃ©menter
   Ã  l'aveugle : 1 h de test avant 2 jours de code.
2. Si nÃ©cessaire : parser incrÃ©mental â€” accumuler les deltas, dÃ©tecter <tool_call>
   en cours de flux, Ã©mettre les chunks OpenAI `delta.tool_calls[...]` conformes
   (index, id, name puis arguments fragmentÃ©s), finir par finish_reason:"tool_calls".
3. Cas tordu Ã  couvrir : le tag <tool_call> coupÃ© entre deux chunks du modÃ¨le.
**Acceptation** : un client openai officiel en mode stream reÃ§oit un tool_call
complet et valide ; test unitaire du tag coupÃ© entre chunks.

### C3 â€” Plusieurs tool-calls par rÃ©ponse
Les agents demandent souvent 2-3 outils d'un coup (lire fichier A + fichier B).
1. Ã‰tendre core/tool_calls.py : parser N blocs <tool_call> successifs â†’ liste
   tool_calls[] avec index corrects.
2. Test : prompt qui induit naturellement 2 appels ("compare weather in Paris and Tokyo",
   tools=[get_weather]) â†’ 2 tool_calls dans la rÃ©ponse.
**Acceptation** : test unitaire multi-calls 5/5 + le test e2e Ã  2 appels passe.

### C4 â€” Contexte long rÃ©aliste (T6.4 du plan, version minimale)
Un agent envoie 30-120K tokens. Il faut savoir ce que TA machine tient VRAIMENT.
1. Mesurer sur le serveur Qwen3.6-35B-A3B : TTFT et tok/s decode Ã  8K / 32K / 64K
   de prompt (gÃ©nÃ©rer des prompts de code synthÃ©tiques), et la VRAM du KV Ã  chaque
   palier. Publier le tableau dans BENCHMARK_RESULTS.md.
2. Fixer le `--ctx` par dÃ©faut du serve Ã  la valeur que la VRAM supporte avec marge
   (pas le max thÃ©orique de 256K du modÃ¨le).
3. RequÃªte dÃ©passant le contexte â†’ erreur 400 propre avec message actionnable
   ("prompt is X tokens, server context is Y; reduce or restart with --ctx"),
   PAS un crash (intÃ©gration auto-heal T7.6).
**Acceptation** : tableau publiÃ© ; la 400 propre est testÃ©e ; le dÃ©faut --ctx est justifiÃ©
par les mesures.

### C5 â€” Robustesse du function calling (les cas sales)
Les modÃ¨les produisent parfois du JSON malformÃ© dans les arguments. Un agent qui
reÃ§oit un tool_call aux arguments invalides plante ou boucle.
1. Validation des arguments : si le JSON du bloc <tool_call> est malformÃ©, tenter
   une rÃ©paration triviale (quotes, virgule finale) ; si irrÃ©parable, NE PAS Ã©mettre
   de tool_call cassÃ© â€” renvoyer le texte brut en content avec un log WARN.
2. Supporter `tool_choice`: "auto" (dÃ©faut), "none" (ignorer les tools), et si peu
   coÃ»teux "required". Documenter ce qui n'est pas supportÃ© (rÃ©ponse 400 explicite,
   pas d'ignorance silencieuse).
3. Alias de modÃ¨le : paramÃ¨tre `--served-model-name` (ou VRM_MODEL_ALIAS) pour que
   les clients configurÃ©s avec un nom arbitraire matchent /v1/models.
4. Mini-suite de rÃ©gression : 10 scÃ©narios tool-calling (nominal, multi, malformÃ©,
   tool_choice=none, erreur outil, boucle, contexte tool long, unicode dans args,
   args vides, fonction inconnue) dans tests/test_tool_calls_regression.py.
**Acceptation** : 10/10 scÃ©narios passent ; aucun tool_call syntaxiquement invalide
ne peut sortir du serveur.

### C6 â€” Documentation + vitrine (transforme le travail en adoption)
1. `docs/coding_agents.md` : configs copiables-collables pour Aider, Cline/Continue,
   Open WebUI (celles VALIDÃ‰ES en C1, pas thÃ©oriques), avec les piÃ¨ges rencontrÃ©s.
2. Section README "Local coding assistant" â‰¤ 30 lignes : 2 commandes (serve + config
   agent), le GIF asciinema de C1, le tableau de perfs de C4.
3. HONNÃŠTETÃ‰ DES CHIFFRES : le +500% de T7.1 (prompt lookup) a Ã©tÃ© mesurÃ© sur le
   backend HF. Le serve Qwen3.6 passe par llama.cpp â†’ NE PAS afficher +500% pour ce
   chemin sans l'avoir mesurÃ© dessus. VÃ©rifier si le serveur llama.cpp de la version
   installÃ©e expose une option de lookup/speculative decoding ; si oui, la benchmarker
   sur un prompt de code et publier LE chiffre de ce chemin-lÃ  ; sinon, le README
   attribue le +500% explicitement au backend HF.
4. SÃ©curitÃ© par dÃ©faut : bind 127.0.0.1 par dÃ©faut (jamais 0.0.0.0 sans flag explicite
   --host), et documenter VRM_API_TOKEN pour toute exposition rÃ©seau.
**Acceptation** : un lecteur qui suit docs/coding_agents.md Ã  la lettre sur machine
propre obtient une session Aider fonctionnelle ; aucun chiffre du README n'est
attribuÃ© Ã  un chemin oÃ¹ il n'a pas Ã©tÃ© mesurÃ©.

### C7 â€” PrÃ©paration du lancement (aprÃ¨s C0-C6 uniquement)
C'est le moment que tout le reste a prÃ©parÃ© :
1. Tag release (v2.0.0 vu la bascule d'identitÃ© "orchestrateur") + CHANGELOG section
   honnÃªte, dans la lignÃ©e de la 1.5.0.
2. LICENSE MIT Ã  la racine si toujours absent (bloquant pour tout lancement).
3. PyPI (T3.4 du plan) : `pip install vramancer` fonctionnel sur machine vierge.
4. Post r/LocalLLaMA, accroche : "Local coding agent on mismatched consumer GPUs
   (3090 + 5070 Ti): Qwen3.6-35B-A3B served with OpenAI-compatible tool calling,
   one command" + le GIF + le tableau de perfs + lien vers les benchmarks honnÃªtes
   (les rapports d'Ã©chec documentÃ©s sont un ARGUMENT, pas une honte â€” les mettre en avant).
5. PrÃ©parer les rÃ©ponses aux objections prÃ©visibles : "pourquoi pas llama.cpp server
   direct ?" â†’ rÃ©ponse honnÃªte : VRAMancer L'UTILISE et ajoute l'orchestration
   (dÃ©tection, split, auto-heal, tool-call parsing validÃ© agent, auto-backend) ;
   "pourquoi pas Ollama ?" â†’ comparer prÃ©cisÃ©ment ce qu'Ollama ne fait pas sur
   GPUs hÃ©tÃ©rogÃ¨nes + le function calling validÃ©.
**Acceptation** : post publiÃ©, premiÃ¨res questions traitÃ©es avec des liens vers la doc.

---

## 3. IdÃ©es additionnelles de l'architecte (Ã  lire, pas toutes Ã  faire)

Ces points sont du conseil, pas des ordres. Ã€ intÃ©grer si le coÃ»t est marginal
pendant les tÃ¢ches ci-dessus, sinon Ã  parquer dans IDEES_PARKING.md.

- **MÃ©triques agent dans /metrics** : compteurs tool_calls Ã©mis / rÃ©ussis / malformÃ©s-rÃ©parÃ©s,
  TTFT et tok/s par requÃªte, taille de prompt. Quand des utilisateurs arriveront,
  ces mÃ©triques diront ce qui casse chez eux. CoÃ»t faible pendant C5.
- **Blocs de raisonnement Qwen3.6** : si le modÃ¨le Ã©met des blocs de thinking, vÃ©rifier
  qu'ils ne fuient pas dans `content` (les agents les prendraient pour la rÃ©ponse).
  Les strip ou les exposer en champ sÃ©parÃ© type `reasoning_content`. Ã€ tester en C1 â€”
  c'est un piÃ¨ge classique des modÃ¨les Qwen rÃ©cents.
- **Gestion de la boucle d'agent longue** : un agent enchaÃ®ne des dizaines de requÃªtes
  qui partagent le mÃªme prÃ©fixe croissant. Le cache de prompt du serveur llama.cpp
  (slots) est le levier â€” vÃ©rifier qu'il est actif et dimensionnÃ© (n_slots, cache rÃ©utilisÃ©),
  c'est la version llama.cpp du T7.0 qui avait Ã©chouÃ© cÃ´tÃ© HF. Gain de TTFT massif
  en usage agent si actif.
- **Timeout adaptatif** : une gÃ©nÃ©ration avec 64K de prefill peut prendre >60 s de TTFT ;
  vÃ©rifier que VRM_GENERATE_TIMEOUT ne tue pas les requÃªtes lÃ©gitimes longues
  (timeout proportionnel Ã  la taille du prompt, ou distinct prefill/decode).
- **Un exemple "zÃ©ro agent"** : script examples/tool_calling_quickstart.py de 30 lignes
  avec le client openai officiel (dÃ©finir un outil, boucle Ã  2 tours). C'est la porte
  d'entrÃ©e des dÃ©veloppeurs qui n'utilisent pas Aider. CoÃ»t : 1 h, valeur doc : haute.
- **Ne PAS faire maintenant** (parking explicite) : multi-modÃ¨les simultanÃ©s, routage
  intelligent local/cloud, RAG intÃ©grÃ©, UI web de chat. Tout Ã§a est sÃ©duisant et tout
  Ã§a diluerait la consolidation. Le projet a une histoire de dispersion ; la
  consolidation est l'anti-dispersion.

## 4. Rappel du cap

La sÃ©quence complÃ¨te est : C0 â†’ C1 â†’ (C2 si nÃ©cessaire) â†’ C3 â†’ C4 â†’ C5 â†’ C6 â†’ C7.
AprÃ¨s C7 seulement : T7.9-Ã©tape1 re-mesurÃ©e sous accelerate, puis l'Ã©tude de
faisabilitÃ© du hook accelerate pour T7.2.

Le critÃ¨re de fin de consolidation tient en une phrase : **un inconnu installe
VRAMancer, suit la doc, et obtient un agent de code local fonctionnel sur ses GPUs
dÃ©pareillÃ©s â€” sans nous parler.** Tout ce qui sert cette phrase est prioritaire ;
tout le reste attend.

â€” L'architecte
