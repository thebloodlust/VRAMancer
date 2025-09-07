# API
## `core.monitor.GPUMonitor`
- `gpus : list[dict]`  
  Liste de dictionnaires contenant `index`, `name`, `type`, `total_memory`, `device`.
- `memory_allocated(gpu_index) -> int`  
  Retourne la mémoire allouée en bytes pour `gpu_index`.

## `core.scheduler.SimpleScheduler`
- `__init__(blocks)`  
  `blocks` est une liste de modules PyTorch.
- `forward(x)` → logits  
  Forward‑pass sur tous les blocs.
- `predict(x)` → predicted indices  
  Retourne la séquence de tokens prédites.
