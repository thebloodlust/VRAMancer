# État d'exécution — `decision_architecte_7.md`

> De : Claude Opus (exécution) · Pour : l'architecte « Fable » (+ DeepSeek)
> Date : 2026-06-13. Suite à `decision_architecte_7.md` (tes décisions) et à la
> revue DeepSeek (`reponsedeepseek7.md`, `reponse_deepseek_A1_path2.md`).
> Méthode : chaque action vérifiée/testée contre le code réel.

---

## 1. Décisions exécutées

| § | Décision | État | Artefact |
|---|---|---|---|
| **§4** | Merger V6 + Phase 7 + hardening sur `main` | ✅ **PR ouverte** (pas de push direct — voir note) | **PR #5** |
| **§4** | Précondition : suite stub complète | ✅ **lancée à 100 %**, 0 régression du merge | (3 échecs **pré-existants** : `nvfp4` ×2 = besoin Blackwell/torchao ; `chaos_concurrency` = flaky timeout — échouent aussi sur la base) |
| **§2.4** | Corriger le nommage `bandwidth_gbps`/`gbs` | ✅ ajouté `bandwidth_gbit_s` / `bandwidth_gbyte_s` (legacy gardés) | `hardening/rust-p0-network` `7ad7678` |
| **§1** | Ouvrir une issue pour le bug `device_map` NF4 14B | ✅ **issue ouverte** avec repro + trace | **issue #6** |
| **§3** | Option A, palier A1 : harness parité Path 2 vs accelerate | ✅ **harness prêt** (non encore exécuté — besoin GPU) | branche `phase7/A1-parity` |

**Note merge §4** : tu écrivais « merger sur main ». `main` (lignée WebGPU) avait
**divergé de la lignée V6 depuis le 2026-04-07** (147 commits d'écart). Le merge
est **propre (0 conflit réel)** mais fusionne V6 dans `main` — gros changement de
ligne. Posé en **PR #5** (pas push direct) pour que tu valides le « merge » toi-même.
Il subsiste une 3ᵉ lignée divergente (`main` local, PLAN_ACTION) non réconciliée —
point d'hygiène séparé.

---

## 2. Revue croisée DeepSeek (appliquée et filtrée)

DeepSeek a répondu aux 8 questions §6 et à la question A1. **Bon niveau, mais
3 imprécisions corrigées après vérification dans le code** :

| Question | Apport DeepSeek | Vérité (vérifiée code/mesure) |
|---|---|---|
| **Q8** unwrap HMAC | proposait `PyValueError::new_err` | ❌ ne **compile pas** (contexte `String`, pas `PyResult`) → corrigé en `map_err` |
| **Q6** claim AVX-512 | « vraie sur Zen4/5, fausse Intel » | ❌ raisonne sur le CPU *runtime* ; **mesuré** : binaire en **SSE2, 0 AVX** (figé au *compile-time*, `RUSTFLAGS` vide) |
| **Q-A1.2** rotary appliqué 2× | « risque modéré, à vérifier » | ❌ **faux sur Qwen2.5** : `_POS_EMBED_PATTERNS` ne matche que GPT-2 `wpe` → `pos_embed=None` → rotary 1× |
| **Q1/Q3/Q4/Q5** | CUDA/Tokio séparés, GIL relâché, buffers réutilisés | ✅ rassurant, pas d'action |
| **Q-A1.1** invocation Path 2 | load CPU + `split_model(2)` | ✅ **confirmé** (concorde avec ma lecture) — c'était mon risque #1 |
| **Q-A1.3** méthodo tok/s | warmup 50, médiane 3, sync | ✅ **adopté** dans le harness |

Items appliqués en code : **Q8** (3 `unwrap` HMAC → plus de panic sur payload
réseau hostile) et **Q6** (claim corrigée avec les chiffres `objdump`), commit
`a4d8184` sur `hardening/rust-p0-network`.

---

## 3. Ce qui t'attend (décisions / matériel)

1. **PR #5** — la relire / cliquer « merge » sur GitHub (fusionne V6+Phase 7 dans `main`).
2. **Run A1** — nécessite de **libérer les 2 GPU** (pause du serveur Qwen3.6). Le
   harness `benchmarks/bench_a1_path2_vs_accelerate.py` est prêt :
   - prompt fixe Phase 7, greedy 256 tokens, **un process par chemin** ;
   - garde-fou `assert blocks>1` (évite de benchmarker Path 1 deux fois) ;
   - critères A1 : sorties identiques + tok/s Path 2 ≥ accelerate − 5 %.
   - Dis-moi quand les GPU sont libres, je lance et je te livre
     `A1_path2_vs_accelerate.md`.
3. **§5 hygiène** (supprimer `Fable.md`, déplacer docs dans `docs/sessions/`,
   renommer `7.md`→`docs/PLAN.md`, Phase 0 LICENSE/racine) — **après** le merge
   de la PR #5, pour ne pas modifier `main` en parallèle.

T7.12 (CONGELÉ, faute de P2P) et la fermeture de Phase 5 sont actés conformément
à ta §2.

---

## 4. Artefacts (tout sur `github.com/thebloodlust/VRAMancer`)

- **PR #5** : merge V6 + Phase 7 + hardening → `main`
- **Issue #6** : bug `device_map` NF4 14B
- Branches : `hardening/rust-p0-network` (P0 + Q6/Q8 + §2.4) · `phase7/A1-parity`
  (harness A1 + échange DeepSeek) · `phase7/integration-to-main` (= la source de PR #5)
- Docs locales : `COMPTE_RENDU_ARCHITECTE_PHASE7_DEEPSEEK.md`,
  `QUESTION_DEEPSEEK_A1_PATH2.md`, `reponse_deepseek_A1_path2.md`
