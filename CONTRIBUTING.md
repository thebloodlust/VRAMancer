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
```

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
