# IDÉES PARKING — hors périmètre consolidation (règle Fable, 2026-07-05)

> Règle : pendant la consolidation C0-C7, AUCUNE feature hors liste. Toute idée
> séduisante atterrit ici, pas dans le code. À reconsidérer APRÈS C7.

## Parkées explicitement par l'architecte (§3, ne PAS faire maintenant)
- Multi-modèles simultanés.
- Routage intelligent local/cloud.
- RAG intégré.
- UI web de chat.

## Conseils de l'architecte à intégrer SI coût marginal (sinon ici)
- Métriques agent dans `/metrics` (tool_calls émis/réussis/malformés, TTFT, tok/s, taille prompt) — coût faible pendant C5.
- Blocs de raisonnement Qwen3.6 : vérifier qu'ils ne fuient pas dans `content` (strip ou champ `reasoning_content`) — à tester en C1.
- Cache de prompt llama.cpp (slots `n_slots`) : gros gain TTFT en usage agent — vérifier actif/dimensionné.
- Timeout adaptatif (proportionnel au prefill) : ne pas tuer les longues requêtes légitimes.
- `examples/tool_calling_quickstart.py` (30 l., client openai officiel, boucle 2 tours) — porte d'entrée dev.

## Idées d'Opus/DeepSeek (sessions précédentes), gelées
- Cross-vendor NVIDIA+AMD (attend GPU AMD) · cross-nœud Thunderbolt (attend 2e machine).
- Placement asymétrique FP4/BF16 (petite expérience, après consolidation).
