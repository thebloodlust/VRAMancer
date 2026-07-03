# MIGRATION_NOTES — journal de refactor (règle R8 de l'architecte)

Suivi des déplacements post-A1 (branche `chore/hygiene-post-a1`). Règles : R1 (rien
supprimé, tout déplacé), R2 (`git mv`), R6 (1 commit/tâche).

## Phase hygiène (§5 update architecte 2026-07-03)

### T-hyg-1 — Échanges de session → `docs/sessions/`
Déplacé tous les comptes-rendus/échanges Opus↔DeepSeek↔architecte de la racine vers
`docs/sessions/` (fichiers `reponse_*.md`, `DEEPSEEK_*.md`, analyses AMD, `SUMMARY_PHASE2-6.md`).
Non référencés par le README → aucun lien cassé.

### T-hyg-2 — Docs utilisateur → `docs/`
`FEATURES.md`, `EXAMPLES.md`, `README_FACILE.md`, `INSTALL_MAC.md`, `INSTALL_WINDOWS.md`,
`TODO.md`, `ROADMAP_PERFORMANCES.md` → `docs/`. (Fusion en un seul `docs/ROADMAP.md`
laissée à l'architecte pour éviter toute perte de contenu.)

### Restent à la racine (canoniques)
`README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `BENCHMARK_RESULTS.md`, `RESULTAT_PALIER_A1.md`,
`LICENSE`, `MIGRATION_NOTES.md`.

Tests avant/après : voir commits.
