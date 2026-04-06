# Installation VRAMancer MLX Worker — MacBook M4

## Étape 1 : Créer un venv arm64 et installer les dépendances

**IMPORTANT** : MLX nécessite un Python **arm64 natif** (pas Rosetta/x86).

Même si tu installes Python 3.14 arm64 depuis python.org, le `python3` dans le PATH peut encore pointer vers une version x86 (Homebrew x86, ancienne install, etc.).

**Vérifier l'architecture :**
```bash
python3 -c "import platform; print(platform.machine())"
```
→ Doit afficher `arm64`. Si ça affiche `x86_64`, il faut utiliser le **chemin complet** du Python arm64.

**Trouver le bon Python arm64 :**
```bash
# Python installé depuis python.org (3.14)
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 -c "import platform; print(platform.machine())"

# Python Homebrew arm64 (si installé via /opt/homebrew)
/opt/homebrew/bin/python3 -c "import platform; print(platform.machine())"

# Chercher tous les Python disponibles
find /Library/Frameworks /opt/homebrew/bin /usr/local/bin -name "python3*" 2>/dev/null
```

**Créer le venv avec le bon Python (celui qui affiche arm64) :**
```bash
# Exemple avec Python 3.14 installé depuis python.org :
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 -m venv ~/venv_vrm

# OU avec Homebrew arm64 :
# /opt/homebrew/bin/python3 -m venv ~/venv_vrm

# Activer et vérifier que c'est bien arm64
source ~/venv_vrm/bin/activate
python3 -c "import platform; print(platform.machine())"
# → DOIT afficher arm64

# Installer les dépendances
pip install mlx mlx-lm numpy
```

## Étape 2 : Créer le fichier worker

Créer le fichier `~/mac_worker.py` et coller tout le code ci-dessous :

```bash
cat > ~/mac_worker.py << 'ENDOFFILE'
#!/usr/bin/env python3
"""VTP MLX Compute Worker — runs transformer layers on Apple Silicon via MLX.

Companion to the VRAMancer distributed inference system.
Loads transformer layers from a Qwen/Llama/etc model and processes
hidden states received over the VTP binary protocol.

Requirements (Mac only):
    pip install mlx mlx-lm numpy

Usage:
    python3 mac_worker.py --model mlx-community/Qwen2.5-14B-4bit --start-layer 42 --end-layer 48
    python3 mac_worker.py --model mlx-community/Qwen2.5-7B-4bit --start-layer 20 --end-layer 28

VTP binary protocol:
  Request:  VTP1(4B) | start_layer(H) | end_layer(H) | seq_len(I) |
            ndim(B) | dtype(B) | shape(I*ndim) | payload_len(I) | raw_bytes
  Response: VTP1(4B) | ndim(B) | dtype(B) | shape(I*ndim) | payload_len(I) | raw_bytes
"""

import socket
import struct
import threading
import argparse
import time
import sys
import gc
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    print("ERROR: MLX not found. Install with: pip install mlx mlx-lm numpy")
    sys.exit(1)

VTP_MAGIC = b"VTP1"

# ─── Dtype encoding (matches VRAMancer VTP protocol) ─────────────
DTYPE_NP = {
    0: np.float32, 1: np.float16, 3: np.float64,
    4: np.int32, 5: np.int64, 6: np.int8, 7: np.uint8, 8: np.int16,
}
DTYPE_MX = {
    0: mx.float32, 1: mx.float16, 2: mx.bfloat16, 3: mx.float32,
}
CODE_FROM_MX = {mx.float32: 0, mx.float16: 1, mx.bfloat16: 2}


# ─── Socket helpers ──────────────────────────────────────────────

def recv_exact(sock, n):
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1048576))
        if not chunk:
            raise ConnectionError("closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# ─── BFloat16 conversions (numpy lacks bfloat16) ────────────────

def bf16_bytes_to_f32_np(data, shape):
    u16 = np.frombuffer(data, dtype=np.uint16).reshape(shape).copy()
    f32_bits = u16.astype(np.uint32) << 16
    return f32_bits.view(np.float32)

def f32_np_to_bf16_bytes(f32_np):
    u32 = f32_np.view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


# ─── Raw bytes ↔ mx.array conversion ────────────────────────────

def raw_to_mx_array(data, dtype_code, shape):
    if dtype_code == 2:  # bfloat16
        f32 = bf16_bytes_to_f32_np(data, shape)
        return mx.array(f32).astype(mx.bfloat16)
    np_dt = DTYPE_NP.get(dtype_code, np.float32)
    arr = np.frombuffer(data, dtype=np_dt).reshape(shape).copy()
    return mx.array(arr)

def mx_array_to_raw(arr, target_dtype_code):
    if target_dtype_code == 2:  # keep bfloat16
        f32 = np.array(arr.astype(mx.float32))
        return f32_np_to_bf16_bytes(f32), 2
    np_arr = np.array(arr.astype(mx.float16))
    return np_arr.tobytes(), 1


# ─── Model state ─────────────────────────────────────────────────
_model = None
_arch = None
_start = 0
_end = 0
_num_layers = 0

def get_layers():
    if _arch == "gpt2":
        return _model.transformer.h
    return _model.model.layers

def load_model(model_name, start_layer, end_layer):
    global _model, _arch, _start, _end, _num_layers
    from mlx_lm import load as mlx_load

    print(f"Loading {model_name} ...")
    t0 = time.time()
    model, _ = mlx_load(model_name)
    elapsed = time.time() - t0
    print(f"Model loaded in {elapsed:.1f}s")

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        _arch = "llama"
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        _arch = "gpt2"
    else:
        print(f"ERROR: Unknown architecture: {type(model)}")
        sys.exit(1)

    _num_layers = len(layers)
    end_layer = min(end_layer, _num_layers)
    print(f"Architecture: {_arch}, {_num_layers} layers, keeping {start_layer}-{end_layer - 1}")

    freed = 0
    for i in range(_num_layers):
        if i < start_layer or i >= end_layer:
            layers[i] = None
            freed += 1

    if _arch == "llama":
        model.model.embed_tokens = None
        if hasattr(model.model, "norm"):
            model.model.norm = None
    elif _arch == "gpt2":
        model.transformer.wte = None
        model.transformer.wpe = None
        if hasattr(model.transformer, "ln_f"):
            model.transformer.ln_f = None
    if hasattr(model, "lm_head"):
        model.lm_head = None

    gc.collect()
    if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()

    print(f"Freed {freed} unused layers + embed/norm/head")
    _model = model
    _start = start_layer
    _end = end_layer
    return end_layer


def create_causal_mask(seq_len, dtype):
    if seq_len <= 1:
        return None
    try:
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        return mask.astype(dtype)
    except Exception:
        indices = mx.arange(seq_len)
        is_masked = indices[:, None] < indices[None, :]
        neginf = mx.array(float("-inf"), dtype=dtype)
        zero = mx.array(0.0, dtype=dtype)
        return mx.where(is_masked, neginf, zero)


def forward_layers(hidden, seq_len):
    layers = get_layers()
    mask = create_causal_mask(seq_len, hidden.dtype)
    for i in range(_start, _end):
        layer = layers[i]
        try:
            out = layer(hidden, mask=mask, cache=None)
        except TypeError:
            try:
                out = layer(hidden, mask=mask)
            except TypeError:
                out = layer(hidden)
        hidden = out[0] if isinstance(out, tuple) else out
    mx.eval(hidden)
    return hidden


# ─── VTP connection handler ──────────────────────────────────────

def handle_conn(conn, addr):
    print(f"[+] Connection from {addr}")
    total_req = 0
    total_ms = 0.0
    t0 = time.monotonic()
    try:
        while True:
            try:
                magic = recv_exact(conn, 4)
            except ConnectionError:
                break
            if magic != VTP_MAGIC:
                print(f"[!] Bad magic: {magic!r}")
                break
            hdr = recv_exact(conn, 10)
            _sl, _el, seq_len, ndim, dtype_code = struct.unpack("!HHIBB", hdr)
            shape_data = recv_exact(conn, ndim * 4)
            shape = struct.unpack(f"!{ndim}I", shape_data)
            plen_data = recv_exact(conn, 4)
            payload_len = struct.unpack("!I", plen_data)[0]
            payload = recv_exact(conn, payload_len)
            total_req += 1

            hidden = raw_to_mx_array(payload, dtype_code, shape)
            t_fwd = time.monotonic()
            result = forward_layers(hidden, seq_len)
            fwd_ms = (time.monotonic() - t_fwd) * 1000
            total_ms += fwd_ms

            if total_req <= 5 or total_req % 10 == 0:
                print(f"  step {total_req}: seq={seq_len} shape={list(shape)} fwd={fwd_ms:.1f}ms")

            out_raw, out_dtype = mx_array_to_raw(result, dtype_code)
            out_shape = tuple(int(s) for s in result.shape)
            out_ndim = len(out_shape)
            resp = VTP_MAGIC
            resp += struct.pack("!BB", out_ndim, out_dtype)
            resp += struct.pack(f"!{out_ndim}I", *out_shape)
            resp += struct.pack("!I", len(out_raw))
            conn.sendall(resp)
            conn.sendall(out_raw)
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        elapsed = time.monotonic() - t0
        if total_req > 0:
            avg = total_ms / total_req
            print(f"[-] {addr}: {total_req} steps, avg fwd={avg:.1f}ms, total={elapsed:.1f}s")
        conn.close()


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VTP MLX Compute Worker for VRAMancer")
    parser.add_argument("--model", required=True,
                        help="MLX model (e.g. mlx-community/Qwen2.5-14B-4bit)")
    parser.add_argument("--start-layer", type=int, required=True)
    parser.add_argument("--end-layer", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18951)
    args = parser.parse_args()

    actual_end = load_model(args.model, args.start_layer, args.end_layer)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    except Exception:
        pass
    srv.bind((args.host, args.port))
    srv.listen(8)

    n_layers = actual_end - args.start_layer
    print(f"\n{'='*50}")
    print(f"  VTP MLX Compute Worker")
    print(f"  Model:   {args.model}")
    print(f"  Layers:  {args.start_layer}-{actual_end - 1} ({n_layers} layers)")
    print(f"  Listen:  {args.host}:{args.port}")
    print(f"  Backend: Apple Silicon MLX (Metal)")
    print(f"{'='*50}")
    print(f"Waiting for connections...\n")

    try:
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
            except Exception:
                pass
            threading.Thread(target=handle_conn, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        srv.close()

if __name__ == "__main__":
    main()
ENDOFFILE
```

## Étape 3 : Lancer le worker

```bash
source ~/venv_vrm/bin/activate
python3 ~/mac_worker.py --model mlx-community/Qwen2.5-14B-4bit --start-layer 42 --end-layer 48
```

Le modèle sera téléchargé automatiquement (~8 GB, une seule fois).

Quand tu vois :
```
Freed 42 unused layers + embed/norm/head

==================================================
  VTP MLX Compute Worker
  Model:   mlx-community/Qwen2.5-14B-4bit
  Layers:  42-47 (6 layers)
  Listen:  0.0.0.0:18951
  Backend: Apple Silicon MLX (Metal)
==================================================
Waiting for connections...
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
