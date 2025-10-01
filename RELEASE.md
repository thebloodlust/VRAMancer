# VRAMancer v0.1.0 — Release Notes

## 🚀 Nouveautés majeures

- **Packaging universel** :
  - Build .deb prêt à l’emploi (voir `build_deb.sh` ou `make deb`)
  - Archive portable `.tar.gz` (`make archive`)
  - Version Lite CLI only (`make lite`)

- **Backends LLM unifiés** :
  - Sélection dynamique : HuggingFace, vLLM, Ollama (auto/CLI/config)
  - Découpage adaptatif selon la VRAM réelle
  - Stubs enrichis pour vLLM/Ollama (prêts pour intégration réelle)

- **Orchestration GPU avancée** :
  - Exploitation automatique des GPU secondaires pour monitoring, offload, orchestration, worker réseau
  - Monitoring GPU secondaire en thread (exemple inclus)

- **Réseau flexible** :
  - Sélection auto/manuel de l’interface réseau (CLI/config)
  - Fallback intelligent si interface indisponible

- **Modularité & Extensibilité** :
  - Dashboards multiples (Qt, Tk, Web, CLI)
  - Premium modules (désactivables en version Lite)
  - Structure Python moderne, packaging pro, tests unitaires

## 🛠️ Installation rapide

```bash
git clone https://github.com/thebloodlust/VRAMancer.git
cd VRAMancer
bash Install.sh
source .venv/bin/activate
make deb           # ou make archive / make lite
```

## 🖥️ Lancement (exemples)

- **Mode auto** (backend, dashboard, GPU, réseau auto)
  ```bash
  python -m vramancer.main
  ```
- **Forcer un backend**
  ```bash
  python -m vramancer.main --backend vllm --model mistral
  ```
- **Version Lite (CLI only)**
  ```bash
  tar -xzf vramancer-lite.tar.gz
  cd vramancer-lite
  python -m vramancer.main --backend huggingface --model gpt2
  ```
- **Sélection réseau manuelle**
  ```bash
  python -m vramancer.main --net-mode manual
  ```

## 🔥 Exploitation GPU secondaires (exemple)

- Les GPU non utilisés pour l’inférence principale sont détectés automatiquement et peuvent être utilisés pour :
  - Monitoring (thread Python inclus)
  - Offload (swap, worker réseau, etc.)

## 🧩 Backends LLM (exemple d’intégration)

```python
from core.backends import select_backend
backend = select_backend("auto")  # ou "huggingface", "vllm", "ollama"
model = backend.load_model("gpt2")
blocks = backend.split_model(num_gpus=2)
out = backend.infer(inputs)
```

## 📝 Changelog

- Packaging .deb/CLI lite universel
- Sélection backend dynamique (auto/CLI/config)
- Découpage adaptatif VRAM
- Usages GPU secondaires (monitoring, offload)
- Sélection réseau auto/manuel
- Backends vLLM/Ollama enrichis
- Version Lite CLI only
- Refactoring, docstring, robustesse accrue

---

**Projet maintenu par thebloodlust — contributions bienvenues !**
