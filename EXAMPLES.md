# Exemples d’utilisation avancée VRAMancer

## 1. Lancement standard (auto)
```bash
python -m vramancer.main
```

## 2. Forcer un backend spécifique
```bash
python -m vramancer.main --backend vllm --model mistral
```

## 3. Version Lite (CLI only)
```bash
tar -xzf vramancer-lite.tar.gz
cd vramancer-lite
python -m vramancer.main --backend huggingface --model gpt2
```

## 4. Sélection réseau manuelle
```bash
python -m vramancer.main --net-mode manual
```

## 5. Exploitation GPU secondaires (monitoring)
- Les GPU secondaires sont détectés automatiquement et monitorés en thread :

```
🔄 GPU secondaires disponibles pour tâches annexes : [1, 2]
[MONITOR] GPU secondaire 1 : 0.0 MB alloués
[MONITOR] GPU secondaire 2 : 0.0 MB alloués
```

## 6. Intégration backend LLM dans un script Python
```python
from core.backends import select_backend
backend = select_backend("auto")  # ou "huggingface", "vllm", "ollama"
model = backend.load_model("gpt2")
blocks = backend.split_model(num_gpus=2)
out = backend.infer(inputs)
```

## 7. Packaging .deb universel
```bash
bash build_deb.sh
sudo dpkg -i vramancer_deb.deb
```

## 8. Archive portable
```bash
make archive
# ou
make lite
```

---

Pour toute question, consultez le README ou ouvrez une issue sur GitHub.
