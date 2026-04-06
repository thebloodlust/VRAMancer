# Installation VRAMancer MLX Worker — MacBook M4

## Étape 1 : Créer un venv et installer les dépendances

```bash
python3 -m venv ~/venv_vrm
source ~/venv_vrm/bin/activate
pip install mlx mlx-lm numpy
```

## Étape 2 : Copier le fichier worker

Copier le fichier `mac` depuis le serveur Ubuntu vers le MacBook :

```bash
scp jeremie@192.168.1.22:/home/jeremie/VRAMancer/VRAMancer/mac ~/mac_worker.py
```

Ou copier-coller le contenu du fichier `mac` dans `~/mac_worker.py` sur le Mac.

## Étape 3 : Lancer le worker

```bash
source ~/venv_vrm/bin/activate
python3 ~/mac_worker.py --model mlx-community/Qwen2.5-14B-4bit --start-layer 42 --end-layer 48
```

Le modèle sera téléchargé automatiquement (~8 GB, une seule fois).

Quand tu vois :
```
Loaded 6 layers (42-47), freed embed/head/unused layers
Listening on 0.0.0.0:18951 — Waiting for connections...
```

C'est prêt. Lance le benchmark depuis Ubuntu.

## Étape 4 : Benchmark (côté Ubuntu)

```bash
cd /home/jeremie/VRAMancer/VRAMancer
source .venv/bin/activate
python benchmarks/bench_3node.py --mac-host 192.168.1.27
```

## Résumé

| Machine | Rôle | Layers |
|---|---|---|
| Ubuntu GPU0 (RTX 3090) | Layers 0-29 + embed/norm/head | 30 |
| Ubuntu GPU1 (RTX 5070 Ti) | Layers 30-41 | 12 |
| MacBook M4 (MLX Metal) | Layers 42-47 (4-bit) | 6 |
