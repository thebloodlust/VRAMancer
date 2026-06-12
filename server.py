#!/usr/bin/env python3
"""VRAMancer — OpenAI-compatible inference server with model management UI.

Endpoints:
  GET  /                        → Web UI (model manager + chat test)
  GET  /health                  → status JSON
  GET  /v1/models               → OpenAI model list
  POST /v1/chat/completions     → OpenAI chat (streaming + non-streaming)
  POST /v1/models/load          → Hot-swap model {"model":"..","num_gpus":2,"quantization":""}
  POST /v1/models/unload        → Unload current model

Usage:
    source .venv/bin/activate
    python server.py                                              # Qwen2.5-14B 2-GPU
    python server.py --model Qwen/Qwen2.5-Coder-32B-Instruct --quantization nvfp4
    python server.py --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --num-gpus 2

VSCode Continue config  ~/.continue/config.json:
    { "models": [{ "title": "VRAMancer", "provider": "openai",
                   "model": "vramancer", "apiBase": "http://localhost:8000",
                   "apiKey": "vramancer" }] }
"""
import os, sys, time, json, uuid, asyncio, argparse, threading, logging, gc
from typing import AsyncIterator, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
log = logging.getLogger("vramancer.server")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("pip install fastapi uvicorn[standard]"); sys.exit(1)

from pydantic import BaseModel, Field

# ── State ─────────────────────────────────────────────────────────────────────
_pipeline      = None   # HuggingFace InferencePipeline
_llama         = None   # LlamaBackend (GGUF)
_vllm          = None   # VLLMBackend
_model_name    = None
_model_loading = threading.Lock()   # one load at a time
_load_status   = {"state": "idle", "progress": "", "error": ""}
_discovery     = None   # ClusterDiscovery instance
_webgpu        = None   # WebGPUBackend instance (for worker list)

def _active_backend():
    """Return whichever backend is currently loaded."""
    if _vllm is not None:   return _vllm
    if _llama is not None:  return _llama
    return _pipeline

# ── Pydantic ──────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "vramancer"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    stream: bool = False
    stop: Optional[list[str]] = None

class LoadRequest(BaseModel):
    model: str
    num_gpus: int = 2
    quantization: str = ""   # "" = bf16, "nvfp4", "nf4", "int8"
    backend: str = "hf"      # "hf" = HuggingFace, "llama" = llama-cpp GGUF, "vllm" = vLLM
    gguf_file: str = ""      # GGUF filename within the HF repo (backend="llama" only)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _messages_to_prompt(messages, tokenizer=None) -> str:
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    parts = []
    for m in messages:
        prefix = {"system": "System", "user": "User", "assistant": "Assistant"}.get(m.role, m.role)
        parts.append(f"{prefix}: {m.content}")
    parts.append("Assistant:")
    return "\n".join(parts)

def _strip_prompt(full: str, prompt: str) -> str:
    return full[len(prompt):] if full.startswith(prompt) else full

def _do_load(model_name: str, num_gpus: int, quantization: str,
             backend: str = "hf", gguf_file: str = ""):
    global _pipeline, _llama, _model_name
    _load_status.update(state="loading", progress=f"Loading {model_name}...", error="")
    try:
        if backend == "llama":
            _do_load_llama(model_name, gguf_file, num_gpus)
        elif backend == "vllm":
            _do_load_vllm(model_name, num_gpus, quantization)
        else:
            _do_load_hf(model_name, num_gpus, quantization)
        _load_status.update(state="ready", progress="", error="")
        log.info("Model ready: %s", model_name)
    except Exception as e:
        _load_status.update(state="error", progress="", error=str(e))
        log.error("Load failed: %s", e)


def _do_load_hf(model_name: str, num_gpus: int, quantization: str):
    global _pipeline, _llama, _model_name
    import os as _os
    if quantization:
        _os.environ["VRM_QUANTIZATION"] = quantization
    else:
        _os.environ.pop("VRM_QUANTIZATION", None)

    from core.inference_pipeline import InferencePipeline
    pipe = InferencePipeline(backend_name="huggingface",
                             enable_metrics=False, enable_discovery=False, verbose=False)
    pipe.load(model_name, num_gpus=num_gpus)

    # Unload any existing backends
    _shutdown_all()
    _pipeline  = pipe
    _model_name = model_name
    try:
        import torch, gc as _gc
        torch.cuda.empty_cache(); _gc.collect()
    except Exception: pass


def _do_load_llama(repo_id: str, gguf_file: str, num_gpus: int):
    global _pipeline, _llama, _model_name
    from core.llama_backend import LlamaBackend, KNOWN_FILES

    filename = gguf_file or KNOWN_FILES.get(repo_id)
    if not filename:
        raise ValueError(
            f"No GGUF filename specified for {repo_id}. "
            "Pass gguf_file or use a known repo."
        )

    _load_status.update(progress=f"Downloading {filename} from {repo_id}…")
    backend = LlamaBackend.from_hub(repo_id, filename, num_gpus=num_gpus, n_ctx=16384)

    _shutdown_all()
    _llama = backend
    _model_name = f"{repo_id}/{filename}"


def _do_load_vllm(model_name: str, num_gpus: int, quantization: str):
    global _pipeline, _llama, _vllm, _model_name
    from core.vllm_backend import VLLMBackend
    _load_status.update(progress=f"vLLM: loading {model_name}…")
    backend = VLLMBackend.from_config(
        model=model_name,
        num_gpus=num_gpus,
        quantization=quantization,
    )
    _shutdown_all()
    _vllm = backend
    _model_name = model_name


def _shutdown_all():
    global _pipeline, _llama, _vllm
    for obj in (_pipeline, _llama, _vllm):
        if obj is not None:
            try: obj.shutdown()
            except Exception: pass
    _pipeline = None
    _llama = None
    _vllm = None

async def _stream_tokens(req: ChatRequest, cid: str) -> AsyncIterator[str]:
    created = int(time.time())
    try:
        if _vllm is not None or _llama is not None:
            # vLLM or llama.cpp — both support native chat completion
            messages = [{"role": m.role, "content": m.content} for m in req.messages]
            backend = _vllm if _vllm is not None else _llama
            gen = backend.chat_stream(messages, max_tokens=req.max_tokens,
                                      temperature=req.temperature, top_p=req.top_p,
                                      top_k=req.top_k)
            for chunk in gen:
                yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':created,'model':_model_name,'choices':[{'index':0,'delta':{'role':'assistant','content':chunk},'finish_reason':None}]})}\n\n"
                await asyncio.sleep(0)
        else:
            # HuggingFace transformers
            from transformers import TextIteratorStreamer
            import torch
            hf      = _pipeline.backend
            model    = hf.model
            tokenizer = hf.tokenizer
            device   = next(model.parameters()).device
            prompt   = _messages_to_prompt(req.messages, tokenizer)
            enc      = tokenizer(prompt, return_tensors="pt")
            ids      = enc["input_ids"].to(device)
            mask     = enc.get("attention_mask")
            if mask is not None: mask = mask.to(device)

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            gkw = dict(input_ids=ids, max_new_tokens=req.max_tokens, streamer=streamer,
                       do_sample=req.temperature != 1.0 or req.top_p != 1.0)
            if mask is not None: gkw["attention_mask"] = mask
            if req.temperature != 1.0: gkw["temperature"] = req.temperature
            if req.top_p != 1.0: gkw["top_p"] = req.top_p

            threading.Thread(target=model.generate, kwargs=gkw, daemon=True).start()
            for chunk in streamer:
                if not chunk: continue
                yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':created,'model':_model_name,'choices':[{'index':0,'delta':{'role':'assistant','content':chunk},'finish_reason':None}]})}\n\n"
                await asyncio.sleep(0)

        yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':created,'model':_model_name,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error':{'message':str(e)}})}\n\n"

# ── App ───────────────────────────────────────────────────────────────────────
try:
    from core import __version__ as _VRM_VERSION
except Exception:
    _VRM_VERSION = "unknown"
app = FastAPI(title="VRAMancer", version=_VRM_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── UI ────────────────────────────────────────────────────────────────────────
_UI_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VRAMancer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh}
.header{background:#161b22;border-bottom:1px solid #30363d;padding:16px 24px;display:flex;align-items:center;gap:12px}
.header h1{font-size:18px;font-weight:600}
.badge{background:#238636;color:#fff;font-size:11px;padding:2px 8px;border-radius:12px}
.badge.loading{background:#d29922}
.badge.error{background:#da3633}
.container{display:grid;grid-template-columns:320px 1fr;gap:0;height:calc(100vh - 57px)}
.sidebar{background:#161b22;border-right:1px solid #30363d;padding:16px;overflow-y:auto}
.sidebar h2{font-size:13px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}
.current-model{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:12px;margin-bottom:16px;font-size:13px}
.current-model .label{color:#8b949e;font-size:11px;margin-bottom:4px}
.current-model .name{color:#58a6ff;word-break:break-all}
.form-group{margin-bottom:12px}
label{display:block;font-size:12px;color:#8b949e;margin-bottom:4px}
input,select{width:100%;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e6edf3;padding:7px 10px;font-size:13px;outline:none}
input:focus,select:focus{border-color:#388bfd}
.btn{width:100%;padding:8px;border:none;border-radius:6px;font-size:13px;font-weight:500;cursor:pointer;transition:.15s}
.btn-primary{background:#238636;color:#fff}
.btn-primary:hover{background:#2ea043}
.btn-danger{background:#da3633;color:#fff;margin-top:8px}
.btn-danger:hover{background:#f85149}
.presets{margin-bottom:16px}
.preset{display:flex;justify-content:space-between;align-items:center;background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px 10px;margin-bottom:6px;cursor:pointer;transition:.1s;font-size:12px}
.preset:hover{border-color:#58a6ff}
.preset .pname{color:#e6edf3;font-weight:500}
.preset .pmeta{color:#8b949e}
.preset .ptag{font-size:10px;padding:1px 6px;border-radius:10px;background:#1f3a5f;color:#58a6ff}
.status-bar{font-size:12px;color:#8b949e;margin-top:12px;padding:8px;background:#0d1117;border-radius:6px;border:1px solid #30363d;min-height:36px}
.chat-area{display:flex;flex-direction:column}
.messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}
.msg{max-width:85%;padding:12px 16px;border-radius:8px;font-size:14px;line-height:1.6;white-space:pre-wrap}
.msg.user{background:#1f3a5f;align-self:flex-end;border-radius:8px 8px 2px 8px}
.msg.assistant{background:#161b22;border:1px solid #30363d;align-self:flex-start;border-radius:8px 8px 8px 2px}
.msg.assistant .thinking{color:#8b949e;font-size:12px;margin-bottom:6px}
.input-area{padding:16px;border-top:1px solid #30363d;display:flex;gap:8px}
.input-area textarea{flex:1;background:#161b22;border:1px solid #30363d;border-radius:6px;color:#e6edf3;padding:10px 14px;font-size:14px;resize:none;outline:none;font-family:inherit}
.input-area textarea:focus{border-color:#388bfd}
.send-btn{background:#238636;border:none;border-radius:6px;color:#fff;padding:10px 20px;cursor:pointer;font-size:14px;white-space:nowrap}
.send-btn:hover{background:#2ea043}
.send-btn:disabled{background:#30363d;cursor:not-allowed}
.gpu-info{margin-top:16px}
.gpu-row{display:flex;justify-content:space-between;font-size:12px;padding:4px 0;border-bottom:1px solid #21262d}
.gpu-row:last-child{border-bottom:none}
.vram-bar{width:100%;height:4px;background:#30363d;border-radius:2px;margin-top:4px}
.vram-fill{height:100%;border-radius:2px;background:#238636;transition:.3s}
.vram-fill.warn{background:#d29922}
.vram-fill.crit{background:#da3633}
</style>
</head>
<body>
<div class="header">
  <h1>VRAMancer</h1>
  <span class="badge" id="status-badge">loading</span>
  <span style="margin-left:auto;font-size:12px;color:#8b949e" id="model-badge">—</span>
</div>
<div class="container">
  <div class="sidebar">
    <h2>Modèle actif</h2>
    <div class="current-model">
      <div class="label">Chargé</div>
      <div class="name" id="current-name">—</div>
    </div>

    <h2>Changer de modèle</h2>
    <div class="presets">
      <div class="preset" onclick="loadPreset('Qwen/Qwen2.5-14B-Instruct','2','')">
        <div><div class="pname">Qwen2.5-14B-Instruct</div><div class="pmeta">~28 GB BF16 · 2 GPUs</div></div>
        <span class="ptag">chat</span>
      </div>
      <div class="preset" onclick="loadPreset('Qwen/Qwen2.5-Coder-32B-Instruct','2','nvfp4')">
        <div><div class="pname">Qwen2.5-Coder-32B</div><div class="pmeta">~16 GB NVFP4 · 2 GPUs</div></div>
        <span class="ptag">code</span>
      </div>
      <div class="preset" onclick="loadPreset('Qwen/Qwen2.5-Coder-14B-Instruct','2','')">
        <div><div class="pname">Qwen2.5-Coder-14B</div><div class="pmeta">~28 GB BF16 · 2 GPUs</div></div>
        <span class="ptag">code</span>
      </div>
      <div class="preset" onclick="loadPreset('deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct','2','')">
        <div><div class="pname">DeepSeek-Coder-V2-Lite</div><div class="pmeta">~32 GB BF16 · 2 GPUs</div></div>
        <span class="ptag">code</span>
      </div>
      <div class="preset" onclick="loadPreset('Qwen/Qwen3-8B','1','')">
        <div><div class="pname">Qwen3-8B</div><div class="pmeta">~16 GB BF16 · 1 GPU</div></div>
        <span class="ptag">chat</span>
      </div>
      <div class="preset" onclick="loadPreset('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B','1','nf4')">
        <div><div class="pname">DeepSeek-R1-Distill-32B</div><div class="pmeta">~18 GB NF4 · 1 GPU (3090) · beats o1-mini</div></div>
        <span class="ptag">code+raisonnement</span>
      </div>
      <div class="preset" onclick="loadPreset('deepseek-ai/DeepSeek-R1-Distill-Qwen-14B','1','')">
        <div><div class="pname">DeepSeek-R1-Distill-14B</div><div class="pmeta">~28 GB BF16 · 1 GPU (3090) · rapide</div></div>
        <span class="ptag">code+raisonnement</span>
      </div>
      <div class="preset" onclick="loadPreset('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B','1','')">
        <div><div class="pname">DeepSeek-R1-Distill-7B</div><div class="pmeta">~14 GB BF16 · 1 GPU · très rapide</div></div>
        <span class="ptag">code</span>
      </div>
      <div class="preset" onclick="loadPresetVLLM('RedHatAI/Qwen3-Coder-Next-NVFP4','2','nvfp4')" style="border-color:#4a2d7a">
        <div><div class="pname">Qwen3-Coder-Next NVFP4</div><div class="pmeta">~40 GB · 2 GPUs · 5070 Ti HW FP4 · batching continu</div></div>
        <span class="ptag" style="background:#2a1a4a;color:#a371f7">vLLM</span>
      </div>
      <div class="preset" onclick="loadPresetVLLM('Qwen/Qwen2.5-Coder-32B-Instruct','2','nvfp4')" style="border-color:#4a2d7a">
        <div><div class="pname">Qwen2.5-Coder-32B NVFP4</div><div class="pmeta">~16 GB · 2 GPUs · Blackwell FP4</div></div>
        <span class="ptag" style="background:#2a1a4a;color:#a371f7">vLLM</span>
      </div>
      <div class="preset" onclick="loadPresetGGUF('unsloth/Qwen3-Coder-Next-GGUF','2','Qwen3-Coder-Next-UD-Q3_K_XL.gguf')" style="border-color:#2d5a27">
        <div><div class="pname">Qwen3-Coder-Next Q3</div><div class="pmeta">36 GB GGUF · 2 GPUs · 80B/3B actifs · >70% SWE-bench</div></div>
        <span class="ptag" style="background:#1a3a1a;color:#3fb950">GGUF agent</span>
      </div>
      <div class="preset" onclick="loadPresetGGUF('unsloth/Qwen3-Coder-Next-GGUF','1','Qwen3-Coder-Next-UD-TQ1_0.gguf')" style="border-color:#2d5a27">
        <div><div class="pname">Qwen3-Coder-Next TQ1 (léger)</div><div class="pmeta">19 GB GGUF · 1 GPU (3090) · rapide</div></div>
        <span class="ptag" style="background:#1a3a1a;color:#3fb950">GGUF agent</span>
      </div>
    </div>

    <div class="form-group">
      <label>Moteur</label>
      <select id="backend-input" onchange="onBackendChange()">
        <option value="hf">HuggingFace transformers</option>
        <option value="llama">llama.cpp — GGUF (rapide, multi-GPU)</option>
        <option value="vllm">vLLM — batching continu + NVFP4 Blackwell</option>
      </select>
    </div>
    <div class="form-group">
      <label>Modèle (HuggingFace repo)</label>
      <input id="model-input" placeholder="Qwen/Qwen2.5-Coder-32B-Instruct" />
    </div>
    <div class="form-group" id="gguf-file-group" style="display:none">
      <label>Fichier GGUF</label>
      <input id="gguf-file-input" placeholder="model-Q3_K_M.gguf" />
    </div>
    <div class="form-group">
      <label>GPUs</label>
      <select id="gpus-input">
        <option value="1">1 GPU</option>
        <option value="2" selected>2 GPUs</option>
      </select>
    </div>
    <div class="form-group" id="quant-group">
      <label>Quantization</label>
      <select id="quant-input">
        <option value="">BF16 (défaut)</option>
        <option value="nvfp4">NVFP4 (Blackwell ×2 perf)</option>
        <option value="nf4">NF4 (bitsandbytes)</option>
        <option value="int8">INT8 (bitsandbytes)</option>
      </select>
    </div>
    <button class="btn btn-primary" onclick="loadModel()">⬆ Charger</button>
    <button class="btn btn-danger" onclick="unloadModel()">✕ Décharger</button>
    <div class="status-bar" id="load-status">En attente…</div>

    <div class="gpu-info" id="gpu-info"></div>

    <div class="gpu-info" id="cluster-panel">
      <h2 style="font-size:13px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin:16px 0 8px">Cluster</h2>
      <div id="cluster-nodes" style="font-size:12px;color:#8b949e">Recherche de nœuds…</div>
      <div style="margin-top:8px;padding:8px;background:#0d1117;border:1px solid #30363d;border-radius:6px;font-size:11px">
        <div style="color:#8b949e;margin-bottom:4px">Ajouter un appareil (MacBook, téléphone…)</div>
        <div id="webnpu-link" style="color:#58a6ff">Chargement…</div>
        <div style="color:#8b949e;margin-top:4px">Ouvrir ce lien dans Safari/Chrome sur l'appareil</div>
      </div>
    </div>
  </div>

  <div class="chat-area">
    <div class="messages" id="messages">
      <div class="msg assistant">Bonjour ! Modèle chargé et prêt. Pose une question ou charge un nouveau modèle depuis le panneau gauche.</div>
    </div>
    <div class="input-area">
      <textarea id="chat-input" rows="2" placeholder="Message… (Shift+Enter = saut de ligne)" onkeydown="handleKey(event)"></textarea>
      <button class="btn send-btn" id="send-btn" onclick="sendMessage()">Envoyer</button>
    </div>
  </div>
</div>

<script>
const API = '';

async function poll() {
  try {
    const h = await fetch(API+'/health').then(r=>r.json());
    const badge = document.getElementById('status-badge');
    const mb = document.getElementById('model-badge');
    const cn = document.getElementById('current-name');
    const st = document.getElementById('load-status');
    if (h.loaded) {
      badge.textContent = 'ready'; badge.className = 'badge';
      const n = h.model || '—';
      mb.textContent = n.split('/').pop();
      cn.textContent = n;
      st.textContent = '✓ Modèle prêt : ' + n;
    } else {
      badge.textContent = h.load_state || 'idle'; badge.className = 'badge loading';
      mb.textContent = '—'; cn.textContent = '—';
      if (h.load_progress) st.textContent = h.load_progress;
      else if (h.load_error) { st.textContent = '✗ ' + h.load_error; badge.className='badge error'; }
    }
    updateGPU(h.gpus || []);
  } catch(e) {}
}

function updateGPU(gpus) {
  const c = document.getElementById('gpu-info');
  if (!gpus.length) return;
  c.innerHTML = '<h2 style="font-size:13px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin:16px 0 8px">VRAM</h2>' +
    gpus.map(g => {
      const pct = g.total > 0 ? Math.round(g.used/g.total*100) : 0;
      const cls = pct > 90 ? 'crit' : pct > 75 ? 'warn' : '';
      return `<div class="gpu-row"><div>
        <div>${g.name.replace('NVIDIA ','')}</div>
        <div style="color:#8b949e;font-size:11px">${g.used.toFixed(1)} / ${g.total.toFixed(1)} GB (${pct}%)</div>
        <div class="vram-bar"><div class="vram-fill ${cls}" style="width:${pct}%"></div></div>
      </div></div>`;
    }).join('');
}

function onBackendChange() {
  const b = document.getElementById('backend-input').value;
  document.getElementById('gguf-file-group').style.display = b === 'llama' ? '' : 'none';
  document.getElementById('quant-group').style.display = b === 'llama' ? 'none' : '';
}

function loadPreset(model, gpus, quant) {
  document.getElementById('backend-input').value = 'hf';
  document.getElementById('model-input').value = model;
  document.getElementById('gpus-input').value = gpus;
  document.getElementById('quant-input').value = quant;
  onBackendChange();
}

function loadPresetGGUF(repo, gpus, ggufFile) {
  document.getElementById('backend-input').value = 'llama';
  document.getElementById('model-input').value = repo;
  document.getElementById('gpus-input').value = gpus;
  document.getElementById('gguf-file-input').value = ggufFile;
  onBackendChange();
}

function loadPresetVLLM(model, gpus, quant) {
  document.getElementById('backend-input').value = 'vllm';
  document.getElementById('model-input').value = model;
  document.getElementById('gpus-input').value = gpus;
  document.getElementById('quant-input').value = quant;
  onBackendChange();
}

async function loadModel() {
  const model = document.getElementById('model-input').value.trim();
  if (!model) return;
  const num_gpus = parseInt(document.getElementById('gpus-input').value);
  const backend = document.getElementById('backend-input').value;
  const quantization = backend === 'hf' ? document.getElementById('quant-input').value : '';
  const gguf_file = backend === 'llama' ? document.getElementById('gguf-file-input').value.trim() : '';
  document.getElementById('load-status').textContent = 'Chargement en cours…';
  try {
    const r = await fetch(API+'/v1/models/load', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model, num_gpus, quantization, backend, gguf_file})
    }).then(r=>r.json());
    document.getElementById('load-status').textContent = r.message || r.status;
  } catch(e) { document.getElementById('load-status').textContent = '✗ '+e; }
}

async function unloadModel() {
  await fetch(API+'/v1/models/unload', {method:'POST'});
  document.getElementById('load-status').textContent = 'Modèle déchargé.';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function addMsg(role, text) {
  const m = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = text;
  m.appendChild(d);
  m.scrollTop = m.scrollHeight;
  return d;
}

async function sendMessage() {
  const inp = document.getElementById('chat-input');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  addMsg('user', text);
  const btn = document.getElementById('send-btn');
  btn.disabled = true;
  const aDiv = addMsg('assistant', '…');

  try {
    const resp = await fetch(API+'/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        model: 'vramancer',
        messages: [{role:'user', content:text}],
        max_tokens: 512,
        stream: true
      })
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '', content = '';
    aDiv.textContent = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const d = line.slice(6).trim();
        if (d === '[DONE]') break;
        try {
          const j = JSON.parse(d);
          const delta = j.choices?.[0]?.delta?.content || '';
          content += delta;
          aDiv.textContent = content;
          document.getElementById('messages').scrollTop = 999999;
        } catch {}
      }
    }
  } catch(e) { aDiv.textContent = '✗ Erreur: '+e; }
  btn.disabled = false;
}

async function pollCluster() {
  try {
    const c = await fetch(API+'/v1/cluster').then(r=>r.json());
    const el = document.getElementById('cluster-nodes');
    const all = [
      {...c.local, role:'server (local)', online:true},
      ...c.peers
    ];
    if (all.length === 0) { el.innerHTML = '<span style="color:#8b949e">Aucun nœud détecté</span>'; return; }

    el.innerHTML = all.map(n => {
      const vramTotal = (n.gpus||[]).reduce((s,g) => s + (g.total_gb || (g.total_memory||0)/1e9), 0);
      const vramUsed  = (n.gpus||[]).reduce((s,g) => s + (g.used_gb || 0), 0);
      const gpuNames  = (n.gpus||[]).map(g => g.name?.replace('NVIDIA ','').replace('GeForce ','') || '?').join(' + ') || 'GPU';
      const dot = n.online ? '🟢' : '🔴';
      const isLocal = n.role === 'server (local)';
      const isWebNPU = n.type === 'webnpu';
      const tagColor = isLocal ? '#58a6ff' : isWebNPU ? '#3fb950' : '#d29922';
      const tagLabel = isLocal ? 'local' : isWebNPU ? 'WebGPU' : 'VRAMancer';
      const tag = `<span style="font-size:10px;padding:1px 5px;border-radius:8px;background:#0d1117;border:1px solid ${tagColor};color:${tagColor}">${tagLabel}</span>`;
      const extra = isWebNPU && n.ops_done > 0
        ? `<div style="color:#3fb950;font-size:11px">${n.ops_done} ops · ${n.gflops} GFLOPS · ${n.uptime_s}s</div>`
        : '';
      return `<div style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px 10px;margin-bottom:6px">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px">
          <span style="color:#e6edf3;font-weight:500;font-size:12px">${dot} ${n.hostname}</span>
          ${tag}
        </div>
        <div style="color:#8b949e;font-size:11px;margin-top:3px">${gpuNames}</div>
        <div style="color:#8b949e;font-size:11px">${n.ip}</div>
        ${extra}
        ${vramTotal > 0 ? `<div class="vram-bar" style="margin-top:4px"><div class="vram-fill" style="width:${Math.round(vramUsed/vramTotal*100)}%"></div></div>` : ''}
      </div>`;
    }).join('');

    // Hint if only local node
    if (c.peers.length === 0) {
      el.innerHTML += `<div style="color:#8b949e;font-size:11px;margin-top:6px;padding:6px;background:#0d1117;border:1px dashed #30363d;border-radius:6px">
        Pour ajouter un nœud (ex: laptop RTX 4060):<br>
        <code style="color:#58a6ff">python server.py --node</code>
      </div>`;
    }
  } catch(e) {}
}

// Fill WebNPU link with server's LAN IPv4
(async function() {
  try {
    const h = await fetch(API+'/v1/cluster').then(r=>r.json());
    // Prefer IPv4 (no colon = not IPv6)
    let ip = h.local?.ip || window.location.hostname;
    if (ip.includes(':')) ip = window.location.hostname; // fallback if IPv6
    const url = `https://${ip}:8765/webnpu.html`;
    const el = document.getElementById('webnpu-link');
    el.innerHTML = `<a href="${url}" target="_blank" style="color:#58a6ff">${url}</a>`;
  } catch(e) {}
})();

setInterval(poll, 2000);
setInterval(pollCluster, 5000);
poll();
pollCluster();
</script>
</body>
</html>"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def ui():
    return _UI_HTML

@app.get("/health")
async def health():
    gpus = []
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            gpus.append({
                "name": torch.cuda.get_device_name(i),
                "used": round((total - free) / 1e9, 2),
                "total": round(total / 1e9, 2),
            })
    except Exception:
        pass
    return {
        "status": "ok",
        "loaded": _active_backend() is not None,
        "backend": "vllm" if _vllm is not None else ("llama" if _llama is not None else ("hf" if _pipeline is not None else "none")),
        "model": _model_name,
        "load_state": _load_status["state"],
        "load_progress": _load_status["progress"],
        "load_error": _load_status["error"],
        "gpus": gpus,
    }

@app.get("/v1/cluster")
async def cluster_info():
    """Return local node info + all discovered cluster peers."""
    from experimental.cluster_discovery import get_local_info
    local = get_local_info()
    # Enrich with live VRAM usage
    try:
        import torch
        for g in local.get("gpus", []):
            idx = g.get("id", "").replace("cuda:", "")
            if idx.isdigit():
                free, total = torch.cuda.mem_get_info(int(idx))
                g["used_gb"] = round((total - free) / 1e9, 2)
                g["total_gb"] = round(total / 1e9, 2)
    except Exception:
        pass

    local_hostname = local["hostname"]
    local_ip = local["ip"]

    peers = []

    # mDNS/UDP peers — exclude self (same hostname or same IPv4)
    if _discovery is not None:
        for n in _discovery.get_nodes():
            if n.get("hostname") == local_hostname:
                continue
            if n.get("ip") == local_ip:
                continue
            peers.append({
                "hostname": n.get("hostname", "?"),
                "ip": n.get("ip", "?"),
                "gpus": n.get("gpus", []),
                "gpu_count": n.get("gpu_count", 0),
                "platform_type": n.get("platform_type", "unknown"),
                "vramancer_port": n.get("vramancer_port", 5000),
                "online": True,
                "type": "vramancer",
            })

    # WebGPU browser workers (MacBook, phone, etc.)
    if _webgpu is not None:
        try:
            with _webgpu._workers_lock:
                for w in _webgpu._workers:
                    ip = w.addr[0] if w.addr else "?"
                    peers.append({
                        "hostname": f"browser@{ip}",
                        "ip": ip,
                        "gpus": [{"name": f"WebGPU ({w.backend if hasattr(w,'backend') else 'GPU'})",
                                  "total_gb": 0}],
                        "gpu_count": 1,
                        "platform_type": "browser",
                        "ops_done": w.ops_done,
                        "gflops": round(w.total_gflops, 1),
                        "uptime_s": round(__import__('time').monotonic() - w.connected_at),
                        "online": True,
                        "type": "webnpu",
                    })
        except Exception:
            pass

    return {
        "local": {
            "hostname": local_hostname,
            "ip": local_ip,
            "gpus": local.get("gpus", []),
            "gpu_count": local.get("gpu_count", 0),
            "role": "server",
        },
        "peers": peers,
        "total_vram_gb": sum(
            g.get("total_gb", g.get("total_memory", 0) / 1e9)
            for g in local.get("gpus", [])
        ),
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": _model_name or "vramancer",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "vramancer",
        }],
    }

@app.post("/v1/models/load")
async def api_load(req: LoadRequest):
    if not _model_loading.acquire(blocking=False):
        raise HTTPException(409, "Another model is currently loading")
    def _run():
        try:
            _do_load(req.model, req.num_gpus, req.quantization,
                     backend=req.backend, gguf_file=req.gguf_file)
        finally:
            _model_loading.release()
    threading.Thread(target=_run, daemon=True).start()
    return {"status": "loading", "message": f"Loading {req.model}… check /health for progress"}

@app.post("/v1/models/unload")
async def api_unload():
    global _model_name
    _shutdown_all()
    _model_name = None
    try:
        import torch, gc
        torch.cuda.empty_cache(); gc.collect()
    except Exception: pass
    return {"status": "unloaded"}

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if _active_backend() is None:
        raise HTTPException(503, detail="No model loaded. Use POST /v1/models/load")

    cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if req.stream:
        return StreamingResponse(
            _stream_tokens(req, cid),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    t0 = time.perf_counter()
    if _vllm is not None or _llama is not None:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        backend = _vllm if _vllm is not None else _llama
        answer = backend.chat(messages, max_tokens=req.max_tokens,
                              temperature=req.temperature, top_p=req.top_p, top_k=req.top_k)
    else:
        tokenizer = getattr(_pipeline.backend, "tokenizer", None)
        prompt = _messages_to_prompt(req.messages, tokenizer)
        text = _pipeline.generate(prompt, max_new_tokens=req.max_tokens,
                                   temperature=req.temperature, top_p=req.top_p, top_k=req.top_k)
        answer = _strip_prompt(text, prompt)
    elapsed = time.perf_counter() - t0

    return {
        "id": cid, "object": "chat.completion", "created": int(time.time()),
        "model": _model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": -1, "completion_tokens": len(answer.split()),
                  "total_tokens": -1},
        "vramancer": {"elapsed_s": round(elapsed, 3)},
    }

# ── Startup ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--quantization", default="")
    parser.add_argument("--backend", default="hf", choices=["hf", "llama"])
    parser.add_argument("--gguf-file", default="")
    parser.add_argument("--no-load", action="store_true", help="Start without loading a model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model = args.model or ("" if args.no_load else "Qwen/Qwen2.5-14B-Instruct")
    if model:
        threading.Thread(
            target=_do_load,
            args=(model, args.num_gpus, args.quantization),
            kwargs={"backend": args.backend, "gguf_file": args.gguf_file},
            daemon=True,
        ).start()

    # Start WebGPU/WebNPU WebSocket server (port 8765) for browser workers
    try:
        import os as _os
        # SSL activé — WebNN/WebGPU requièrent un secure context (HTTPS/WSS)
        # Le navigateur affichera un avertissement cert auto-signé à accepter une fois
        from core.webgpu_backend import WebGPUBackend
        _webgpu = WebGPUBackend(ws_host="0.0.0.0", ws_port=8765, serve_ui=True)
        log.info("WebNPU server → https://%s:8765/webnpu.html", args.host)
    except Exception as e:
        log.warning("WebNPU server unavailable: %s", e)

    # Start cluster discovery
    global _discovery
    try:
        from experimental.cluster_discovery import ClusterDiscovery
        _discovery = ClusterDiscovery(port=args.port)
        _discovery.start()
        log.info("Cluster discovery started (mDNS port=%d)", args.port)
    except Exception as e:
        log.warning("Cluster discovery unavailable: %s", e)

    log.info("VRAMancer server → http://%s:%d", args.host, args.port)
    log.info("UI → http://localhost:%d/", args.port)
    log.info("VSCode Continue: apiBase=http://localhost:%d  apiKey=vramancer", args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
