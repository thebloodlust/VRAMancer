# Continuous batching

VRAMancer intègre un batcher continu (style vLLM/Orca) qui regroupe les requêtes concurrentes en un seul forward pass.

## Activation

```bash
export VRM_CONTINUOUS_BATCHING=1
python server.py --model gpt2
```

## Paramètres

| Variable | Défaut | Rôle |
|---|---|---|
| `VRM_CONTINUOUS_BATCHING` | `0` | `1` active le batcher |
| `VRM_GENERATE_TIMEOUT` | `300` | Timeout (secondes) par requête |
| `VRM_MAX_BATCH_SIZE` | `32` | Taille de batch max simultanée |

File d'attente max : **256 requêtes**. Au-delà, `submit()` est bloquant ou rejette.

## Limites

- **GIL CPython** : la tokenization Python reste sérialisée. Speedup au-delà de ~4 requêtes simultanées.
- **KV cache scope** : chaque requête a son propre KV cache paged. Pas de prefix caching actuellement.
- **Backends** : HuggingFace Transformers uniquement. vLLM et llama-cpp ont leur propre batching natif.

## Diagnostic

`InferencePipeline.batcher_stats()` expose : `pending_requests`, `running_batch_size`, `tokens_per_second`, `queue_depth`.
