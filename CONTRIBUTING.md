# Contribuer à VRAMancer

Merci de votre intérêt pour VRAMancer ! Ce guide explique comment contribuer efficacement au projet.

## 🚦 Pré-requis
- Python 3.9+
- `git`, `make`, `pip`
- (optionnel) GPU CUDA/ROCm/MPS pour tests avancés

## 🛠️ Installation dev
```bash
git clone https://github.com/thebloodlust/VRAMancer.git
cd VRAMancer
bash Install.sh
source .venv/bin/activate
make dev-install
```

## 🧪 Tests & lint
```bash
make test
make lint
make coverage # Génère les rapports de couverture (HTML/Terminal) sans le mode Minimal
```

### Stratégie de couverture (V9)
En mode `VRM_MINIMAL_TEST=1` (mode de test par défaut et CI rapide), la couverture peut paraître artificiellement basse (~15%) car les backends (VLLM, Ollama, TensorRT) et les moteurs lourds (compute, network fastpath) sont stubbés pour accélérer l'exécution de la suite de tests et ne pas nécessiter de GPU physique.

Pour visualiser la couverture plus réaliste en développement local :
1. Les modules lourdement *stubés* (`core/backends_vllm.py`, `core/backends_ollama.py`, `core/compute_engine.py`, etc.) sont exclus via `pyproject.toml` (section `[tool.coverage.run] omit`).
2. Lancez `make coverage` afin de s'affranchir du contexte de tests CI stricts et générer un rapport sur les fichiers de logiques orchestrales (`inference_pipeline.py`, `scheduler.py`). Un seuil sain attendu se situe entre 40 et 60%.

## 💡 Proposer une contribution
1. Forkez le repo et créez une branche (`feature/ma-fonctionnalite`)
2. Codez, testez, documentez
3. Vérifiez : `make test` et `make lint` doivent passer
4. Ouvrez une Pull Request claire (FR ou EN)

## 📦 Packaging
- `.deb` : `make deb` ou `bash build_deb.sh`
- Archive portable : `make archive`
- Version Lite CLI : `make lite`

## 🤖 Bonnes pratiques
- Respectez la structure modulaire (core/, dashboard/, cli/, ...)
- Privilégiez l’auto-détection, le fallback, la robustesse
- Documentez vos fonctions (docstring, exemples)
- Ajoutez des tests si possible
- Soyez inclusif et bienveillant dans les issues/PR

## 📝 Licences & crédits
- Licence MIT
- Contributions bienvenues, mentionnez les sources si code tiers

---

Pour toute question, ouvrez une issue ou contactez thebloodlust sur GitHub.
