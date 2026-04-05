#!/usr/bin/env python3
"""VTP Multi-Node Inference Demo — split TinyLlama across two GPUs via TCP.

Architecture:
  Node-A (GPU 0): embed_tokens + layers 0..N-1 → VTP TCP → Node-B
  Node-B (GPU 1): layers N..end + norm + lm_head → sample → Node-A
"""
import os, sys, time, threading, queue, copy
sys.path.insert(0, "/home/jeremie/VRAMancer/VRAMancer")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.network.llm_transport import LLMTransport, VTPServer, VTPOpcode

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SPLIT = 11
GPU_A, GPU_B = 0, 1
MAX_TOKENS = 32
PROMPT = "The future of distributed AI inference is"

# ── Load & split model ─────────────────────────────────────────────────
print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).eval()
num_layers = len(model.model.layers)

# Rotary on each GPU (before splitting — deepcopy from CPU)
rotary_a = copy.deepcopy(model.model.rotary_emb).to(f"cuda:{GPU_A}")
rotary_b = copy.deepcopy(model.model.rotary_emb).to(f"cuda:{GPU_B}")

# Split
model.model.embed_tokens = model.model.embed_tokens.to(f"cuda:{GPU_A}")
for i in range(SPLIT):
    model.model.layers[i] = model.model.layers[i].to(f"cuda:{GPU_A}")
for i in range(SPLIT, num_layers):
    model.model.layers[i] = model.model.layers[i].to(f"cuda:{GPU_B}")
model.model.norm = model.model.norm.to(f"cuda:{GPU_B}")
model.lm_head = model.lm_head.to(f"cuda:{GPU_B}")

torch.cuda.synchronize(GPU_A)
torch.cuda.synchronize(GPU_B)
vram_a = torch.cuda.memory_allocated(GPU_A) / 1024**3
vram_b = torch.cuda.memory_allocated(GPU_B) / 1024**3
print(f"  Split: layers 0-{SPLIT-1} GPU{GPU_A} ({vram_a:.2f}GB), "
      f"layers {SPLIT}-{num_layers-1} GPU{GPU_B} ({vram_b:.2f}GB)")

# ── VTP Transport ──────────────────────────────────────────────────────
transport_b = LLMTransport(node_id="node-B")
server = VTPServer(transport_b, host="0.0.0.0", port=0)
port = server.start()
time.sleep(0.3)
transport_a = LLMTransport(node_id="node-A")
transport_a.connect_peer_tcp("node-B", "127.0.0.1", port)
time.sleep(0.5)
print(f"  VTP: node-A → node-B (port {port}, {transport_a.tier.name})")

# ── Node-B worker ──────────────────────────────────────────────────────
activation_q = queue.Queue()
token_q = queue.Queue()

def node_b():
    while True:
        msg = activation_q.get()
        if msg is None:
            break
        step, hs_cpu, seq_len = msg
        t0 = time.perf_counter()
        with torch.no_grad():
            hs = hs_cpu.to(f"cuda:{GPU_B}")
            pos = torch.arange(seq_len, device=f"cuda:{GPU_B}").unsqueeze(0)
            mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=f"cuda:{GPU_B}"),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0).half()
            pe = rotary_b(hs, pos)
            for i in range(SPLIT, num_layers):
                out = model.model.layers[i](hs, attention_mask=mask,
                                            position_ids=pos, position_embeddings=pe)
                hs = out[0]
                if hs.ndim == 2:
                    hs = hs.unsqueeze(0)
            hs = model.model.norm(hs)
            logits = model.lm_head(hs)
            tok = logits[:, -1, :].argmax(dim=-1)
        torch.cuda.synchronize(GPU_B)
        token_q.put((tok.cpu(), time.perf_counter() - t0))

threading.Thread(target=node_b, daemon=True).start()

# ── Generate ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"VTP Multi-Node Inference")
print(f"  Model: {MODEL} ({num_layers} layers)")
print(f"  Split: 0-{SPLIT-1} → GPU{GPU_A} | {SPLIT}-{num_layers-1} → GPU{GPU_B}")
print(f"  Transport: VTP/TCP (loopback)")
print(f"  Prompt: \"{PROMPT}\"")
print(f"{'='*60}\n")

input_ids = tokenizer.encode(PROMPT, return_tensors="pt")
tokens = []
t_a_total = t_vtp_total = t_b_total = 0.0
vtp_bytes = 0

for step in range(MAX_TOKENS):
    ids = input_ids.to(f"cuda:{GPU_A}")
    seq_len = ids.shape[1]

    # ── Node A: embed + first layers ──
    t0 = time.perf_counter()
    with torch.no_grad():
        hs = model.model.embed_tokens(ids)
        pos = torch.arange(seq_len, device=f"cuda:{GPU_A}").unsqueeze(0)
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=f"cuda:{GPU_A}"),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0).half()
        pe = rotary_a(hs, pos)
        for i in range(SPLIT):
            out = model.model.layers[i](hs, attention_mask=mask,
                                        position_ids=pos, position_embeddings=pe)
            hs = out[0]
            if hs.ndim == 2:
                hs = hs.unsqueeze(0)
    torch.cuda.synchronize(GPU_A)
    dt_a = time.perf_counter() - t0
    t_a_total += dt_a

    # ── VTP send activation ──
    t1 = time.perf_counter()
    nb = hs.nelement() * hs.element_size()
    vtp_bytes += nb
    transport_a.send_tensor(hs, "node-B", dst_gpu=GPU_B, layer_id=step)
    activation_q.put((step, hs.cpu(), seq_len))
    dt_vtp = time.perf_counter() - t1
    t_vtp_total += dt_vtp

    # ── Wait for Node B ──
    tok_cpu, dt_b = token_q.get()
    t_b_total += dt_b
    tid = tok_cpu.item()
    tokens.append(tid)
    input_ids = torch.cat([input_ids, tok_cpu.unsqueeze(0)], dim=1)

    dt_total = dt_a + dt_vtp + dt_b
    if step < 5 or step == MAX_TOKENS - 1:
        bw = nb / dt_vtp / 1e9 if dt_vtp > 0 else 0
        print(f"  Step {step:2d}: A={dt_a*1000:5.1f}ms "
              f"VTP={dt_vtp*1000:4.1f}ms({nb/1024:4.0f}KB {bw:.1f}GB/s) "
              f"B={dt_b*1000:5.1f}ms "
              f"tok={repr(tokenizer.decode([tid]))}")
    elif step == 5:
        print("  ...")

# ── Results ─────────────────────────────────────────────────────────────
activation_q.put(None)
out_text = tokenizer.decode(tokens, skip_special_tokens=True)
total = t_a_total + t_vtp_total + t_b_total

print(f"\n{'='*60}")
print(f"Output: \"{PROMPT}{out_text}\"")
print(f"{'='*60}")
print(f"\n  Node-A: {t_a_total*1000:.0f}ms ({t_a_total/MAX_TOKENS*1000:.1f}ms/step)")
print(f"  VTP:    {t_vtp_total*1000:.0f}ms ({t_vtp_total/MAX_TOKENS*1000:.1f}ms/step, "
      f"{vtp_bytes/1024/1024:.1f}MB, "
      f"{vtp_bytes/t_vtp_total/1e9:.2f} GB/s)")
print(f"  Node-B: {t_b_total*1000:.0f}ms ({t_b_total/MAX_TOKENS*1000:.1f}ms/step)")
print(f"  Total:  {total*1000:.0f}ms = {MAX_TOKENS/total:.1f} tok/s")

stats = transport_a.stats()
print(f"\n  VTP sent: {stats['tensors_sent']} tensors, "
      f"{stats['bytes_sent']/1024/1024:.1f}MB, "
      f"avg {stats['avg_latency_us']:.0f}us")

server.stop()
transport_a.close()
transport_b.close()
