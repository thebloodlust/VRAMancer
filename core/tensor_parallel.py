"""VRAMancer Tensor Parallelism — single-process multi-GPU.

Splits model weights across GPUs (by attention heads / FFN columns)
and uses NCCL all-reduce for synchronisation. Works in a single
process without ``torch.distributed`` — uses ``torch.cuda.nccl``
directly.

Supports:
  - Llama / Mistral / Qwen (separate Q/K/V/O projections)
  - GPT-2 (merged c_attn Conv1D)
  - GQA (grouped-query attention)

Usage::

    from core.tensor_parallel import apply_tensor_parallel

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tp_model = apply_tensor_parallel(model, devices=["cuda:0", "cuda:1"])
    # tp_model.forward() runs both GPUs in parallel with all-reduce
"""

from __future__ import annotations

import os
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger("vramancer.tensor_parallel")

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    torch = None  # type: ignore
    nn = None
    _TORCH = False


# ---------------------------------------------------------------------------
# All-reduce primitive
# ---------------------------------------------------------------------------

def _apply_rotary_pos_emb(x: Any, cos: Any, sin: Any) -> Any:
    """Apply Rotary Position Embedding (RoPE) to query or key tensor.

    x: (batch, heads, seq_len, head_dim)
    cos, sin: (1, 1, seq_len, head_dim) or broadcastable
    """
    # Ensure cos/sin are broadcastable to x shape
    if cos.dim() == 2:  # (seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:  # (1, seq_len, head_dim)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    # Rotate half: split into two halves and apply rotation
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


def _nccl_all_reduce(tensors: List[Any]) -> None:
    """In-place NCCL all-reduce sum (single-process, multi-GPU)."""
    try:
        torch.cuda.nccl.all_reduce(tensors)
    except Exception:
        # Fallback: manual staging via CPU (inference-only, no gradient).
        # sum() would create a compute graph that breaks backprop,
        # but TP is inference-only in VRAMancer so detach is safe.
        with torch.no_grad():
            total = torch.zeros_like(tensors[0], device="cpu")
            for t in tensors:
                total += t.detach().to("cpu")
            for t in tensors:
                t.copy_(total.to(t.device))


def _all_reduce_sum(tensors: List[Any]) -> Any:
    """All-reduce sum and return result on the first device."""
    _nccl_all_reduce(tensors)
    return tensors[0]


# ---------------------------------------------------------------------------
# Weight sharding helpers
# ---------------------------------------------------------------------------

def _shard_column(weight: Any, num_shards: int, dim: int = 0) -> List[Any]:
    """Split weight along output dimension (column-parallel).

    For nn.Linear: weight is [out_features, in_features].
    Column-parallel splits out_features → each GPU gets out_features/N rows.
    """
    return list(weight.chunk(num_shards, dim=dim))


def _shard_row(weight: Any, num_shards: int, dim: int = 1) -> List[Any]:
    """Split weight along input dimension (row-parallel).

    For nn.Linear: weight is [out_features, in_features].
    Row-parallel splits in_features → each GPU gets in_features/N columns.
    """
    return list(weight.chunk(num_shards, dim=dim))


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def _detect_architecture(model: Any) -> str:
    """Detect model architecture from config or module names."""
    config = getattr(model, "config", None)
    arch = getattr(config, "model_type", "").lower() if config else ""

    if arch in ("llama", "mistral", "qwen2", "gemma", "phi3"):
        return "llama"

    if arch in ("gpt2", "gpt_neo", "gpt_neox", "opt"):
        return "gpt2"

    # Check for GPT-2 Conv1D pattern
    for name, _ in model.named_modules():
        if "c_attn" in name:
            return "gpt2"
        if "q_proj" in name:
            return "llama"

    _logger.warning("Unknown architecture %r — defaulting to llama TP sharding; "
                     "this may produce wrong shapes for unsupported models.", arch)
    return "llama"  # default assumption


def _get_layers(model: Any) -> Tuple[Any, List[Any], str]:
    """Extract transformer layers from a HuggingFace model.

    Returns (model_base, layers_list, architecture).
    """
    arch = _detect_architecture(model)

    # Llama / Mistral: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, list(model.model.layers), arch

    # GPT-2: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer, list(model.transformer.h), arch

    raise ValueError(f"Cannot extract layers from {type(model).__name__}")


# ---------------------------------------------------------------------------
# TP layer wrappers
# ---------------------------------------------------------------------------

class TPLinear(nn.Module):
    """Tensor-parallel linear layer shard (runs on one GPU)."""

    def __init__(self, weight: Any, bias: Optional[Any], device: str):
        super().__init__()
        self.weight = nn.Parameter(weight.to(device), requires_grad=False)
        self.bias = nn.Parameter(bias.to(device), requires_grad=False) if bias is not None else None

    def forward(self, x: Any) -> Any:
        return nn.functional.linear(x, self.weight, self.bias)


class TPAttention(nn.Module):
    """Tensor-parallel attention: Q/K/V column-split, O row-split + all-reduce.

    Each GPU handles num_heads/tp_degree attention heads independently.
    After the output projection (row-parallel), an all-reduce sums partials.
    """

    def __init__(
        self,
        original_attn: Any,
        devices: List[str],
        arch: str,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
    ):
        super().__init__()
        self.devices = devices
        self.tp = len(devices)
        self.arch = arch
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.heads_per_shard = num_heads // self.tp
        self.kv_heads_per_shard = max(1, num_kv_heads // self.tp)

        if arch == "gpt2":
            self._init_gpt2(original_attn)
        else:
            self._init_llama(original_attn)

    def _init_llama(self, attn: Any) -> None:
        """Shard Llama-style separate Q/K/V/O projections."""
        q_w = attn.q_proj.weight.data
        k_w = attn.k_proj.weight.data
        v_w = attn.v_proj.weight.data
        o_w = attn.o_proj.weight.data

        q_b = getattr(attn.q_proj, "bias", None)
        k_b = getattr(attn.k_proj, "bias", None)
        v_b = getattr(attn.v_proj, "bias", None)
        o_b = getattr(attn.o_proj, "bias", None)
        q_b = q_b.data if q_b is not None else None
        k_b = k_b.data if k_b is not None else None
        v_b = v_b.data if v_b is not None else None
        o_b = o_b.data if o_b is not None else None

        # Column-parallel: split output dim (rows in weight matrix)
        q_shards = _shard_column(q_w, self.tp)
        k_shards = _shard_column(k_w, self.tp)
        v_shards = _shard_column(v_w, self.tp)

        q_b_shards = _shard_column(q_b, self.tp) if q_b is not None else [None] * self.tp
        k_b_shards = _shard_column(k_b, self.tp) if k_b is not None else [None] * self.tp
        v_b_shards = _shard_column(v_b, self.tp) if v_b is not None else [None] * self.tp

        # Row-parallel: split input dim (columns in weight matrix)
        o_shards = _shard_row(o_w, self.tp)
        o_b_shards = [o_b if i == 0 else None for i in range(self.tp)] if o_b is not None else [None] * self.tp

        self.q_shards = nn.ModuleList([TPLinear(q_shards[i], q_b_shards[i], self.devices[i]) for i in range(self.tp)])
        self.k_shards = nn.ModuleList([TPLinear(k_shards[i], k_b_shards[i], self.devices[i]) for i in range(self.tp)])
        self.v_shards = nn.ModuleList([TPLinear(v_shards[i], v_b_shards[i], self.devices[i]) for i in range(self.tp)])
        self.o_shards = nn.ModuleList([TPLinear(o_shards[i], o_b_shards[i], self.devices[i]) for i in range(self.tp)])

    def _init_gpt2(self, attn: Any) -> None:
        """Shard GPT-2 merged c_attn (Conv1D: [in, 3*out]) + c_proj."""
        w = attn.c_attn.weight.data   # [in_features, 3*hidden]
        b = attn.c_attn.bias.data if hasattr(attn.c_attn, "bias") else None

        hidden = w.shape[1] // 3
        q_w = w[:, :hidden].t()         # -> [hidden, in]
        k_w = w[:, hidden:2*hidden].t()
        v_w = w[:, 2*hidden:].t()

        q_b = b[:hidden] if b is not None else None
        k_b = b[hidden:2*hidden] if b is not None else None
        v_b = b[2*hidden:] if b is not None else None

        q_shards = _shard_column(q_w, self.tp)
        k_shards = _shard_column(k_w, self.tp)
        v_shards = _shard_column(v_w, self.tp)
        q_b_shards = _shard_column(q_b, self.tp) if q_b is not None else [None] * self.tp
        k_b_shards = _shard_column(k_b, self.tp) if k_b is not None else [None] * self.tp
        v_b_shards = _shard_column(v_b, self.tp) if v_b is not None else [None] * self.tp

        o_w = attn.c_proj.weight.data.t()  # Conv1D -> [out, in]
        o_b = attn.c_proj.bias.data if hasattr(attn.c_proj, "bias") else None
        o_shards = _shard_row(o_w, self.tp)
        o_b_shards = [o_b if i == 0 else None for i in range(self.tp)] if o_b is not None else [None] * self.tp

        self.q_shards = nn.ModuleList([TPLinear(q_shards[i], q_b_shards[i], self.devices[i]) for i in range(self.tp)])
        self.k_shards = nn.ModuleList([TPLinear(k_shards[i], k_b_shards[i], self.devices[i]) for i in range(self.tp)])
        self.v_shards = nn.ModuleList([TPLinear(v_shards[i], v_b_shards[i], self.devices[i]) for i in range(self.tp)])
        self.o_shards = nn.ModuleList([TPLinear(o_shards[i], o_b_shards[i], self.devices[i]) for i in range(self.tp)])

    def forward(self, hidden_states: Any, **kwargs) -> Any:
        """Parallel attention across all GPUs + all-reduce."""
        position_embeddings = kwargs.get("position_embeddings")
        partials = []
        for i, dev in enumerate(self.devices):
            h = hidden_states.to(dev, non_blocking=True)
            q = self.q_shards[i](h)
            k = self.k_shards[i](h)
            v = self.v_shards[i](h)

            # Reshape for attention
            bsz, seq_len, _ = q.shape
            q = q.view(bsz, seq_len, self.heads_per_shard, self.head_dim).transpose(1, 2)
            kv_heads = self.kv_heads_per_shard
            k = k.view(bsz, seq_len, kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, seq_len, kv_heads, self.head_dim).transpose(1, 2)

            # Apply rotary position embeddings (RoPE) for Llama/Qwen/Mistral
            if position_embeddings is not None and self.arch != "gpt2":
                cos, sin = position_embeddings
                cos = cos.to(dev)
                sin = sin.to(dev)
                q = _apply_rotary_pos_emb(q, cos, sin)
                k = _apply_rotary_pos_emb(k, cos, sin)

            # GQA expand — handle non-divisible head counts
            if kv_heads < self.heads_per_shard:
                if self.heads_per_shard % kv_heads == 0:
                    repeat = self.heads_per_shard // kv_heads
                    k = k.repeat_interleave(repeat, dim=1)
                    v = v.repeat_interleave(repeat, dim=1)
                else:
                    # Non-divisible: expand then slice to target size
                    repeat = (self.heads_per_shard + kv_heads - 1) // kv_heads
                    k = k.repeat_interleave(repeat, dim=1)[:, :self.heads_per_shard]
                    v = v.repeat_interleave(repeat, dim=1)[:, :self.heads_per_shard]

            # Scaled dot-product attention
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=(seq_len > 1))
            attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)

            # Row-parallel output projection
            out = self.o_shards[i](attn_out)
            partials.append(out)

        # All-reduce sum across GPUs
        _nccl_all_reduce(partials)
        return partials[0]


class TPMLP(nn.Module):
    """Tensor-parallel MLP: gate/up column-split, down row-split + all-reduce."""

    def __init__(self, original_mlp: Any, devices: List[str], arch: str):
        super().__init__()
        self.devices = devices
        self.tp = len(devices)
        self.arch = arch

        if arch == "gpt2":
            self._init_gpt2(original_mlp)
        else:
            self._init_llama(original_mlp)

    def _init_llama(self, mlp: Any) -> None:
        """Shard Llama gate_proj + up_proj (column) + down_proj (row)."""
        gate_w = mlp.gate_proj.weight.data
        up_w = mlp.up_proj.weight.data
        down_w = mlp.down_proj.weight.data

        gate_shards = _shard_column(gate_w, self.tp)
        up_shards = _shard_column(up_w, self.tp)
        down_shards = _shard_row(down_w, self.tp)

        self.gate_shards = nn.ModuleList([TPLinear(gate_shards[i], None, self.devices[i]) for i in range(self.tp)])
        self.up_shards = nn.ModuleList([TPLinear(up_shards[i], None, self.devices[i]) for i in range(self.tp)])
        self.down_shards = nn.ModuleList([TPLinear(down_shards[i], None, self.devices[i]) for i in range(self.tp)])

        # Activation function
        act_name = "silu"  # default for Llama
        if hasattr(mlp, "act_fn"):
            act_name = getattr(mlp.act_fn, "__name__", "silu")
        self.act = getattr(torch.nn.functional, act_name, torch.nn.functional.silu)

    def _init_gpt2(self, mlp: Any) -> None:
        """Shard GPT-2 c_fc (column) + c_proj (row)."""
        fc_w = mlp.c_fc.weight.data.t()     # Conv1D [in, out] -> [out, in]
        fc_b = mlp.c_fc.bias.data if hasattr(mlp.c_fc, "bias") else None
        proj_w = mlp.c_proj.weight.data.t()  # Conv1D [in, out] -> [out, in]
        proj_b = mlp.c_proj.bias.data if hasattr(mlp.c_proj, "bias") else None

        fc_shards = _shard_column(fc_w, self.tp)
        fc_b_shards = _shard_column(fc_b, self.tp) if fc_b is not None else [None] * self.tp
        proj_shards = _shard_row(proj_w, self.tp)
        proj_b_shards = [proj_b if i == 0 else None for i in range(self.tp)] if proj_b is not None else [None] * self.tp

        self.gate_shards = nn.ModuleList([TPLinear(fc_shards[i], fc_b_shards[i], self.devices[i]) for i in range(self.tp)])
        self.up_shards = None  # GPT-2 has no gate/up split
        self.down_shards = nn.ModuleList([TPLinear(proj_shards[i], proj_b_shards[i], self.devices[i]) for i in range(self.tp)])
        self.act = torch.nn.functional.gelu

    def forward(self, hidden_states: Any) -> Any:
        partials = []
        for i, dev in enumerate(self.devices):
            h = hidden_states.to(dev, non_blocking=True)
            if self.up_shards is not None:
                # Llama-style: SiLU(gate(x)) * up(x)
                gate_out = self.act(self.gate_shards[i](h))
                up_out = self.up_shards[i](h)
                intermediate = gate_out * up_out
            else:
                # GPT-2-style: GELU(fc(x))
                intermediate = self.act(self.gate_shards[i](h))
            out = self.down_shards[i](intermediate)
            partials.append(out)

        _nccl_all_reduce(partials)
        return partials[0]


class TPTransformerBlock(nn.Module):
    """One transformer layer running in tensor-parallel mode.

    LayerNorms are replicated on all devices (cheap, needs full hidden_state).
    Attention and MLP are sharded across GPUs with all-reduce after each.
    """

    def __init__(
        self,
        original_block: Any,
        devices: List[str],
        arch: str,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
    ):
        super().__init__()
        self.devices = devices
        self.primary = devices[0]

        # Layer norms: replicate on primary device only (executed before sharding)
        if arch == "gpt2":
            self.ln_1 = original_block.ln_1.to(self.primary)
            self.ln_2 = original_block.ln_2.to(self.primary)
            attn_mod = original_block.attn
            mlp_mod = original_block.mlp
        else:
            self.ln_1 = original_block.input_layernorm.to(self.primary)
            self.ln_2 = original_block.post_attention_layernorm.to(self.primary)
            attn_mod = original_block.self_attn
            mlp_mod = original_block.mlp

        self.tp_attn = TPAttention(attn_mod, devices, arch, head_dim, num_heads, num_kv_heads)
        self.tp_mlp = TPMLP(mlp_mod, devices, arch)

    def forward(self, hidden_states: Any, **kwargs) -> Any:
        # Ensure on primary for layernorm
        h = hidden_states.to(self.primary)

        # Pre-norm attention
        residual = h
        h = self.ln_1(h)
        # Forward position info to attention (critical for Llama/Qwen RoPE)
        attn_kwargs = {}
        for k in ("position_ids", "position_embeddings", "attention_mask",
                   "past_key_value", "cache_position"):
            if k in kwargs:
                attn_kwargs[k] = kwargs[k]
        h = self.tp_attn(h, **attn_kwargs)
        h = residual + h.to(self.primary)

        # Pre-norm MLP
        residual = h
        h = self.ln_2(h)
        h = self.tp_mlp(h)
        h = residual + h.to(self.primary)

        return h


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TPModel(nn.Module):
    """Tensor-parallel wrapper for a HuggingFace CausalLM model.

    Replaces transformer layers with TP-sharded versions.
    Embedding and LM head remain on the primary device.
    """

    def __init__(self, model: Any, devices: List[str]):
        super().__init__()
        self.devices = devices
        self.primary = devices[0]

        model_base, layers, arch = _get_layers(model)
        config = model.config

        num_heads = getattr(config, "num_attention_heads", 12)
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        hidden_size = getattr(config, "hidden_size", 768)
        head_dim = hidden_size // num_heads

        _logger.info(
            "TP init: %d layers, %d heads (%d KV), head_dim=%d, tp=%d, arch=%s",
            len(layers), num_heads, num_kv_heads, head_dim, len(devices), arch,
        )

        # Embeddings on primary
        if arch == "gpt2":
            self.embed = model.transformer.wte.to(self.primary)
            self.pos_embed = model.transformer.wpe.to(self.primary)
            self.final_norm = model.transformer.ln_f.to(self.primary)
        else:
            self.embed = model_base.embed_tokens.to(self.primary)
            self.pos_embed = None
            self.final_norm = model_base.norm.to(self.primary)

        self.lm_head = model.lm_head.to(self.primary)

        # Rotary embeddings (Llama/Mistral)
        self.rotary_emb = None
        if hasattr(model_base, "rotary_emb"):
            self.rotary_emb = model_base.rotary_emb.to(self.primary)

        # TP transformer layers
        self.tp_layers = nn.ModuleList()
        for layer in layers:
            tp_block = TPTransformerBlock(
                layer, devices, arch, head_dim, num_heads, num_kv_heads,
            )
            self.tp_layers.append(tp_block)

        self.arch = arch
        self.config = config
        self._original_model = model  # keep ref for config access

        _logger.info("TP model ready: %d TP layers across %s", len(self.tp_layers), devices)

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        **kwargs,
    ) -> Any:
        device = self.primary
        input_ids = input_ids.to(device)
        seq_len = input_ids.shape[1]

        # Embedding
        h = self.embed(input_ids)
        if self.pos_embed is not None:
            # GPT-2: absolute position embeddings
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            h = h + self.pos_embed(position_ids.to(device))

        # Compute rotary position embeddings for Llama/Qwen/Mistral
        position_embeddings = None
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            position_embeddings = self.rotary_emb(h, position_ids.to(device))

        # Build kwargs to pass through to each layer
        layer_kwargs = {}
        if position_ids is not None:
            layer_kwargs["position_ids"] = position_ids.to(device)
        if position_embeddings is not None:
            layer_kwargs["position_embeddings"] = position_embeddings
        if attention_mask is not None:
            layer_kwargs["attention_mask"] = attention_mask.to(device)

        # Transformer layers
        for layer in self.tp_layers:
            h = layer(h, **layer_kwargs)

        # Final norm + LM head
        h = self.final_norm(h)
        logits = self.lm_head(h)

        return logits

    def generate_greedy(self, input_ids: Any, max_new_tokens: int = 128) -> Any:
        """Simple greedy generation loop for TP model."""
        device = self.primary
        generated = input_ids.to(device)
        seq_len = generated.shape[1]

        for step in range(max_new_tokens):
            # Pass full sequence with correct position_ids
            pos_ids = torch.arange(generated.shape[1], device=device).unsqueeze(0)
            logits = self.forward(generated, position_ids=pos_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            # Check EOS
            eos_id = getattr(self.config, "eos_token_id", None)
            if eos_id is not None and next_token.item() == eos_id:
                break

        return generated


def apply_tensor_parallel(model: Any, devices: List[str] = None) -> TPModel:
    """Convert a HuggingFace CausalLM model to tensor-parallel.

    Args:
        model: A loaded HuggingFace model (AutoModelForCausalLM).
        devices: List of CUDA devices, e.g. ["cuda:0", "cuda:1"].
                 Defaults to all available GPUs.

    Returns:
        TPModel wrapping the sharded weights.
    """
    if devices is None:
        try:
            from core.utils import detect_backend
            backend = detect_backend()
        except ImportError:
            backend = "cuda"
        if backend in ("cuda", "rocm"):
            # ROCm uses the torch.cuda API
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        elif backend == "mps":
            # MPS is single-device; TP not applicable
            raise ValueError("Tensor parallelism is not supported on MPS (single device)")
        else:
            raise ValueError(f"Tensor parallelism requires CUDA/ROCm GPUs, got backend={backend!r}")

    if len(devices) < 2:
        raise ValueError("Tensor parallelism requires at least 2 devices")

    return TPModel(model, devices)


__all__ = [
    "apply_tensor_parallel",
    "TPModel",
    "TPTransformerBlock",
    "TPAttention",
    "TPMLP",
]
