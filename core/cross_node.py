"""
Cross-node distributed inference via HTTP layer relay.

Enables pipeline-parallel inference across multiple VRAMancer nodes
on a LAN.  Each node executes a range of transformer layers and relays
hidden states to the next node via HTTP.

Master:  POST /api/distributed/generate  (orchestrates the generation)
Worker:  POST /api/worker/forward_layers  (runs assigned layers)
"""

import io
import time
import struct
from typing import List, Tuple, Optional

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from core.logger import get_logger

logger = get_logger("cross_node")


# ─── Tensor serialisation (torch.save/load, all dtypes) ──────────

def tensor_to_bytes(t: "torch.Tensor") -> bytes:
    buf = io.BytesIO()
    torch.save(t.detach().cpu(), buf)
    return buf.getvalue()


def bytes_to_tensor(data: bytes, device: str = "cpu") -> "torch.Tensor":
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=True, map_location="cpu").to(device)


# ─── Model architecture helpers ──────────────────────────────────

def get_model_layers(model):
    """Return the nn.ModuleList of transformer layers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers          # Llama / Qwen / Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h         # GPT-2
    raise ValueError(f"Unknown architecture: {type(model)}")


def _is_gpt2(model) -> bool:
    return hasattr(model, "transformer") and hasattr(model.transformer, "h")


def get_model_info(model) -> dict:
    layers = get_model_layers(model)
    return {
        "num_layers": len(layers),
        "arch": "gpt2" if _is_gpt2(model) else "llama",
        "hidden_size": getattr(model.config, "hidden_size", 0),
    }


# ─── Worker-side: run a range of layers ──────────────────────────

def worker_forward(model, hidden_bytes: bytes, start_layer: int, end_layer: int,
                   seq_len: int = 0) -> bytes:
    """Execute layers [start_layer, end_layer) on *hidden_bytes*.

    Called on the worker node.  Computes position_ids & rotary embeddings
    internally when needed (Llama/Qwen).
    """
    device = str(next(model.parameters()).device)
    hidden = bytes_to_tensor(hidden_bytes, device)
    layers = get_model_layers(model)
    gpt2 = _is_gpt2(model)

    # Position IDs & rotary (Llama / Qwen only)
    position_ids = None
    position_embeddings = None
    if not gpt2 and seq_len > 0:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
            try:
                pe = model.model.rotary_emb(hidden, position_ids)
                position_embeddings = tuple(t.to(device) for t in pe)
            except Exception:
                pass

    with torch.no_grad():
        for i in range(start_layer, end_layer):
            kwargs = {}
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            try:
                out = layers[i](hidden, **kwargs)
            except TypeError:
                # Fallback: architecture does not accept these kwargs
                out = layers[i](hidden)
            hidden = out[0] if isinstance(out, tuple) else out

    return tensor_to_bytes(hidden)


# ─── Master-side: proxy to a remote worker ────────────────────────

class RemoteWorker:
    """HTTP proxy that forwards hidden-state tensors to a remote VRAMancer node."""

    def __init__(self, url: str, token: str, start_layer: int, end_layer: int):
        self.url = url.rstrip("/")
        self.start_layer = start_layer
        self.end_layer = end_layer
        self._sess = _requests.Session()
        self._sess.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        })

    def forward(self, hidden_states: "torch.Tensor",
                seq_len: int = 0) -> "torch.Tensor":
        """POST hidden states to worker, receive processed tensor back."""
        payload = tensor_to_bytes(hidden_states)
        resp = self._sess.post(
            f"{self.url}/api/worker/forward_layers",
            params={
                "start_layer": self.start_layer,
                "end_layer": self.end_layer,
                "seq_len": seq_len,
            },
            data=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return bytes_to_tensor(resp.content)


# ─── Master-side: distributed generation ─────────────────────────

def distributed_generate(
    backend,
    prompt: str,
    remote_workers: List[RemoteWorker],
    local_layer_range: Tuple[int, int],
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> dict:
    """Token-by-token generation with layers split across nodes.

    No KV-cache (full recompute per step).  Suitable for demo
    and moderate-length generations.

    Returns
    -------
    dict with keys: text, tokens, total_seconds, tokens_per_second
    """
    model = backend.model
    tokenizer = backend.tokenizer
    device = str(next(model.parameters()).device)
    gpt2 = _is_gpt2(model)
    layers = get_model_layers(model)

    # Execution plan — sorted by layer index
    segments = [{"type": "local", "start": local_layer_range[0],
                 "end": local_layer_range[1]}]
    for w in remote_workers:
        segments.append({"type": "remote", "start": w.start_layer,
                         "end": w.end_layer, "worker": w})
    segments.sort(key=lambda s: s["start"])

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generated = input_ids.clone()

    t0 = time.perf_counter()

    with torch.no_grad():
        for step in range(max_new_tokens):
            seq_len = generated.shape[1]

            # ── Embedding ─────────────────────────────────────────
            if gpt2:
                pos = torch.arange(seq_len, device=device).unsqueeze(0)
                hidden = model.transformer.wte(generated) + model.transformer.wpe(pos)
                if getattr(model.transformer, "drop", None) is not None:
                    hidden = model.transformer.drop(hidden)
            else:
                hidden = model.model.embed_tokens(generated)

            # Position IDs & rotary for local Llama/Qwen layers
            position_ids = None
            position_embeddings = None
            if not gpt2:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                if hasattr(model.model, "rotary_emb"):
                    try:
                        pe = model.model.rotary_emb(hidden, position_ids)
                        position_embeddings = tuple(t.to(device) for t in pe)
                    except Exception:
                        pass

            # ── Layer segments ────────────────────────────────────
            for seg in segments:
                if seg["type"] == "local":
                    for i in range(seg["start"], seg["end"]):
                        kwargs = {}
                        if position_ids is not None:
                            kwargs["position_ids"] = position_ids
                        if position_embeddings is not None:
                            kwargs["position_embeddings"] = position_embeddings
                        try:
                            out = layers[i](hidden, **kwargs)
                        except TypeError:
                            out = layers[i](hidden)
                        hidden = out[0] if isinstance(out, tuple) else out
                else:
                    # ── Remote forward ────────────────────────────
                    worker = seg["worker"]
                    hidden = worker.forward(hidden, seq_len=seq_len).to(device)

            # ── Norm + LM head ────────────────────────────────────
            if gpt2:
                hidden = model.transformer.ln_f(hidden)
            else:
                norm_dev = str(next(model.model.norm.parameters()).device)
                hidden = model.model.norm(hidden.to(norm_dev))
            head_dev = str(next(model.lm_head.parameters()).device)
            logits = model.lm_head(hidden.to(head_dev))

            # ── Sampling ──────────────────────────────────────────
            next_logits = logits[:, -1, :].float()
            if temperature > 0 and temperature != 1.0:
                next_logits = next_logits / temperature
            if top_k > 0 and top_k < next_logits.size(-1):
                top_vals = torch.topk(next_logits, top_k)[0]
                next_logits[next_logits < top_vals[..., -1:]] = float("-inf")
            probs = torch.softmax(next_logits, dim=-1)
            if top_p < 1.0:
                sorted_p, sorted_i = torch.sort(probs, descending=True)
                cumsum = sorted_p.cumsum(dim=-1)
                sorted_p[(cumsum - sorted_p) >= top_p] = 0.0
                probs = torch.zeros_like(probs).scatter_(-1, sorted_i, sorted_p)
            if temperature > 0:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = probs.argmax(-1, keepdim=True)

            generated = torch.cat([generated, next_token.to(device)], dim=-1)
            if (tokenizer.eos_token_id is not None
                    and next_token.item() == tokenizer.eos_token_id):
                break

    elapsed = time.perf_counter() - t0
    new_tokens = generated[0][input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    n = len(new_tokens)

    return {
        "text": text,
        "tokens": n,
        "total_seconds": round(elapsed, 4),
        "tokens_per_second": round(n / elapsed, 2) if elapsed > 0 else 0,
    }
