"""**Status: demo / local monitoring — not for production deployment.**"""
import os, sys
# Auto-activation du mode minimal APRÈS import os
if os.environ.get('VRM_DASHBOARD_MINIMAL','0') == '0':
    try:
        import importlib
        missing = False
        for mod in ('torch','transformers','numpy'):
            if importlib.util.find_spec(mod) is None:
                missing = True; break
        if missing:
            os.environ['VRM_DASHBOARD_MINIMAL'] = '1'
    except Exception:
        os.environ['VRM_DASHBOARD_MINIMAL'] = '1'
# dashboard/dashboard_web.py

from flask import Flask, render_template, request, jsonify
try:
    from flask_socketio import SocketIO, emit  # type: ignore
except ImportError:  # fallback sans temps réel
    SocketIO = None  # type: ignore
    def emit(*a, **k):
        return None
from math import isnan  # noqa (potentiel usage futur)
try:
    from utils.gpu_utils import get_available_gpus
except Exception:
    def get_available_gpus():
        return []
import subprocess
try:
    from core.security import install_security
except Exception as _sec_err:  # Fallback Windows / extraction incomplète
    def install_security(app):  # type: ignore
        if os.environ.get('VRM_API_DEBUG','0') in {'1','true','TRUE'}:
            print(f"[WARN] security layer indisponible: {_sec_err} -> fallback no-op")

app = Flask(__name__)


import logging
from collections import deque
import time

# Ring buffer in memory to keep the last 200 logs for Supervision
log_buffer = deque(maxlen=200)

class WebsocketLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            log_entry = {
                "time": time.time(),
                "level": record.levelname,
                "module": record.module,
                "msg": msg
            }
            log_buffer.append(log_entry)
            # if socketio is present, broadcast it
            if globals().get('socketio'):
                socketio.emit('new_log', log_entry)
        except Exception:
            pass

try:
    global_logger = logging.getLogger("vramancer")
    ws_handler = WebsocketLogHandler()
    ws_handler.setFormatter(logging.Formatter('%(message)s'))
    global_logger.addHandler(ws_handler)
except Exception:
    pass


@app.route("/api/debug/logs")
def api_debug_logs():
    return jsonify(list(log_buffer))


if SocketIO:
    cors_origins = os.environ.get("VRM_CORS_ORIGINS", "http://localhost:*").split(",")
    if len(cors_origins) == 1 and cors_origins[0] == "*":
        cors_origins = "*"
    socketio = SocketIO(app, cors_allowed_origins=cors_origins)
else:
    socketio = None
install_security(app)

# Objet mémoire hiérarchique injecté dynamiquement par main (set via set_memory_manager)
HIER_MEMORY = None

def set_memory_manager(hm):
    global HIER_MEMORY
    HIER_MEMORY = hm



@app.route("/")
def dashboard():
    gpus = []
    # Try pynvml first for accurate VRAM (includes all allocators, not just torch)
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            gpus.append({
                "name": name,
                "total_vram_mb": info.total // (1024 * 1024),
                "used_vram_mb": info.used // (1024 * 1024),
                "is_available": True,
            })
        pynvml.nvmlShutdown()
    except Exception:
        pass
    # Fallback to torch
    if not gpus:
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        alloc = torch.cuda.memory_allocated(i)
                        total = props.total_memory
                        gpus.append({
                            "name": props.name,
                            "total_vram_mb": total // (1024 * 1024),
                            "used_vram_mb": alloc // (1024 * 1024),
                            "is_available": True,
                        })
                    except Exception:
                        gpus.append({"name": f"GPU {i}", "total_vram_mb": 0, "used_vram_mb": 0, "is_available": False})
        except ImportError:
            pass
    if not gpus:
        gpus = [{"name": "No GPU detected", "total_vram_mb": 0, "used_vram_mb": 0, "is_available": False}]
    memory = None
    return render_template("dashboard.html", gpus=gpus, memory=memory)


@app.route("/api/gpu")
@app.route("/api/dashboard/gpus")
def api_gpu():
    """Real-time GPU info (pynvml preferred, torch fallback)."""
    # Try pynvml first for accurate VRAM
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        devices = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            devices.append({
                "id": i, "name": name,
                "memory_used": info.used, "memory_total": info.total,
                "memory_free": info.free,
                "memory_usage_percent": round((info.used / info.total) * 100, 2) if info.total else 0,
                "gpu_utilization": util.gpu,
            })
        pynvml.nvmlShutdown()
        return jsonify({"cuda_available": True, "device_count": len(devices), "devices": devices})
    except Exception:
        pass
    # Fallback to torch
    try:
        import torch
        if not torch.cuda.is_available():
            return jsonify({"cuda_available": False, "devices": []})
        devices = []
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                alloc = torch.cuda.memory_allocated(i)
                total = props.total_memory
                devices.append({
                    "id": i, "name": props.name,
                    "memory_used": alloc, "memory_total": total,
                    "memory_free": total - alloc,
                    "memory_usage_percent": round((alloc / total) * 100, 2) if total else 0,
                })
            except Exception as e:
                devices.append({"id": i, "error": str(e)})
        return jsonify({"cuda_available": True, "device_count": len(devices), "devices": devices})
    except ImportError:
        return jsonify({"cuda_available": False, "devices": [], "message": "torch not available"})


@app.route("/api/pipeline/status")
def api_pipeline_status():
    """Pipeline status from the global inference pipeline."""
    try:
        from core.inference_pipeline import get_pipeline
        pipe = get_pipeline()
        if pipe and pipe.is_loaded():
            return jsonify(pipe.status())
        return jsonify({"loaded": False, "message": "No model loaded"})
    except Exception as e:
        return jsonify({"loaded": False, "error": str(e)})


CLUSTER_DISCOVERY = None  # injecté par vramancer serve (partage la même discovery)


def set_cluster_discovery(disco):
    """Partage l'instance ClusterDiscovery du serveur avec le dashboard (vue multi-noeuds)."""
    global CLUSTER_DISCOVERY
    CLUSTER_DISCOVERY = disco


@app.route("/api/cluster/nodes")
def api_cluster_nodes():
    """Tous les noeuds du cluster + leurs GPU (vue multi-noeuds)."""
    disco = CLUSTER_DISCOVERY
    try:
        if disco is None:
            # snapshot court si aucune discovery partagée
            import os as _os
            _os.environ.setdefault("VRM_EXPERIMENTAL", "1")
            from experimental.cluster_discovery import ClusterDiscovery
            disco = ClusterDiscovery()
            disco.start()
            import time as _t; _t.sleep(2)
            nodes = list(disco.get_nodes() or [])
            disco.stop()
        else:
            nodes = list(disco.get_nodes() or [])
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e), "nodes": []})
    # Résumé compact par noeud
    out = []
    for n in nodes:
        gpus = n.get("gpus", []) or []
        out.append({
            "hostname": n.get("hostname", "?"), "ip": n.get("ip", "?"),
            "platform": n.get("platform_type", n.get("os", "?")),
            "gpu_count": n.get("gpu_count", len(gpus)),
            "gpus": [{"name": g.get("name", "?"),
                      "vram_gb": round(g.get("total_memory", 0) / 1024**3, 1)} for g in gpus],
            "state": n.get("_state", "?"),
        })
    total_gpus = sum(x["gpu_count"] for x in out)
    return jsonify({"ok": True, "node_count": len(out), "total_gpus": total_gpus, "nodes": out})


@app.route("/cluster")
@app.route("/dash")
def cluster_page():
    """Vue multi-noeuds auto-rafraîchie (page autonome, zéro dépendance)."""
    return """<!doctype html><html><head><meta charset=utf-8><title>VRAMancer Cluster</title>
<style>body{font-family:system-ui,sans-serif;background:#0d1117;color:#c9d1d9;margin:0;padding:24px}
h1{font-size:20px}.node{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin:10px 0}
.host{font-weight:600;color:#58a6ff}.gpu{display:inline-block;background:#21262d;border-radius:6px;padding:6px 10px;margin:4px 6px 0 0;font-size:13px}
.muted{color:#8b949e;font-size:13px}.hdr{display:flex;justify-content:space-between;align-items:center}</style></head>
<body><div class=hdr><h1>🖥️ VRAMancer — Cluster (multi-nœuds)</h1><span class=muted id=sum></span></div>
<div id=nodes></div>
<script>
async function refresh(){
 try{const r=await fetch('/api/cluster/nodes');const d=await r.json();
 document.getElementById('sum').textContent=(d.node_count||0)+' nœud(s) · '+(d.total_gpus||0)+' GPU';
 document.getElementById('nodes').innerHTML=(d.nodes||[]).map(n=>`<div class=node>
   <div class=host>${n.hostname} <span class=muted>${n.ip} · ${n.platform} · ${n.state}</span></div>
   ${(n.gpus||[]).map(g=>`<span class=gpu>${g.name} — ${g.vram_gb} GB</span>`).join('')||'<span class=muted>pas de GPU annoncé</span>'}
 </div>`).join('')||'<p class=muted>Aucun nœud découvert. Lance `vramancer serve` sur d\\'autres machines du LAN.</p>';
 }catch(e){document.getElementById('nodes').innerHTML='<p class=muted>erreur: '+e+'</p>';}
}
refresh();setInterval(refresh,3000);
</script></body></html>"""


def _active_pipeline():
    """Récupère la pipeline active (config dashboard ou globale)."""
    pipe = app.config.get("pipeline")
    if pipe is not None:
        return pipe
    try:
        from core.inference_pipeline import get_pipeline
        return get_pipeline()
    except Exception:
        return None


@app.route("/api/lora/list")
def api_lora_list():
    pipe = _active_pipeline()
    if pipe is None or not getattr(pipe, "is_loaded", lambda: False)():
        return jsonify({"ok": False, "msg": "no model loaded"}), 400
    try:
        return jsonify({"ok": True, **pipe.lora.list()})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/lora/load", methods=["POST"])
def api_lora_load():
    payload = request.get_json(silent=True) or {}
    path = (payload.get("path") or payload.get("adapter") or "").strip()
    name = (payload.get("name") or "").strip() or None
    if not path:
        return jsonify({"ok": False, "msg": "path required"}), 400
    pipe = _active_pipeline()
    if pipe is None:
        return jsonify({"ok": False, "msg": "no model loaded"}), 400
    try:
        return jsonify(pipe.lora.load(path, name))
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/lora/use", methods=["POST"])
def api_lora_use():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "msg": "name required"}), 400
    pipe = _active_pipeline()
    if pipe is None:
        return jsonify({"ok": False, "msg": "no model loaded"}), 400
    try:
        if name.lower() in ("base", "none", "off"):
            return jsonify(pipe.lora.disable())
        return jsonify(pipe.lora.use(name))
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/lora/unload", methods=["POST"])
def api_lora_unload():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    pipe = _active_pipeline()
    if pipe is None or not name:
        return jsonify({"ok": False, "msg": "name required + model loaded"}), 400
    try:
        return jsonify(pipe.lora.unload(name))
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/chat")
def chat_ui():
    return render_template("chat.html")

@app.route("/browser")
def model_browser():
    return render_template("browser.html")

@app.route("/api/models/search")
def api_models_search():
    print(">>> SEARCH ROUTE HIT! Query:", request.args.get("q"))
    try:
        import requests
    except ImportError:
        print(">>> MISSING REQUESTS LIB")
        return jsonify({"results": [{"id": "Erreur: request library missing", "source": "Internal"}]})
    
    query = request.args.get("q", "")
    import urllib.parse
    encoded_query = urllib.parse.quote(query, safe='')
    results = []
    
    # --- Recherche Hugging Face ---
    try:
        print(">>> FETCHING FROM HF...")
        hf_resp = requests.get(f"https://huggingface.co/api/models?search={encoded_query}&limit=10", timeout=3)
        if hf_resp.ok:
            for m in hf_resp.json():
                results.append({"id": m["id"], "source": "Hugging Face"})
        else:
            print(">>> HF ERROR:", hf_resp.status_code, hf_resp.text)
    except Exception as e:
        print(">>> HF EXCEPTION:", e)

    # --- Recherche Ollama locale ---
    try:
        print(">>> FETCHING FROM OLLAMA...")
        ollama_resp = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        if ollama_resp.ok:
            models = ollama_resp.json().get("models", [])
            for m in models:
                if query.lower() in m["name"].lower():
                    results.append({"id": m["name"], "source": "Ollama"})
        else:
            print(">>> OLLAMA ERROR:", ollama_resp.status_code, ollama_resp.text)
    except Exception as e:
        print(">>> OLLAMA EXCEPTION:", e)

    print(">>> RESULTS:", results)
    return jsonify({"results": results})


@app.route("/api/models/load", methods=["POST"])
def api_models_load():
    """Load a HuggingFace model into the active InferencePipeline.

    POST body: {"model_id": "sshleifer/tiny-gpt2", "source": "hf"}
    Returns: {"ok": bool, "msg": str, "model_id": str, "device_map": dict}
    """
    payload = request.get_json(silent=True) or {}
    model_id = payload.get("model_id", "").strip()
    source = payload.get("source", "hf").strip().lower()
    if not model_id:
        return jsonify({"ok": False, "msg": "model_id required"}), 400
    if source not in ("hf", "hugging face", "huggingface"):
        return jsonify({"ok": False, "msg": f"source '{source}' not supported yet"}), 400

    try:
        from core.inference_pipeline import InferencePipeline
        pipeline = app.config.get("pipeline")
        if pipeline is None:
            pipeline = InferencePipeline(enable_metrics=False, enable_discovery=False)
            app.config["pipeline"] = pipeline
        pipeline.load(model_id)
    except Exception as e:
        app.logger.error("Failed to load model %s: %s", model_id, e, exc_info=True)
        return jsonify({"ok": False, "msg": f"load failed: {e}"}), 500

    device_map = getattr(getattr(pipeline, "backend", None), "hf_device_map", None) or {}
    return jsonify({
        "ok": True,
        "msg": f"Loaded {model_id}",
        "model_id": model_id,
        "device_map": dict(device_map) if hasattr(device_map, "items") else {},
    })


@app.route("/api/swarm/status")
def api_swarm_status():
    clients = 0
    flops = 0.0
    tensors = 0
    try:
        from core.metrics import WEBGPU_CONNECTED_CLIENTS, WEBGPU_FLOPS_TOTAL
        clients = max(0, int(WEBGPU_CONNECTED_CLIENTS._value.get()))
        flops = WEBGPU_FLOPS_TOTAL._value.get()
        tensors = int(flops // 15000000) if flops else 0
    except Exception:
        pass

    return jsonify({
        "activeNodes": clients,
        "flopsProcessed": flops,
        "tensorsProcessed": tensors,
        "bandwidth": 0.0
    })

@app.route("/api/memory")
def api_memory():
    if HIER_MEMORY is None:
        return jsonify({"error": "no memory manager"}), 404
    detail = HIER_MEMORY.summary()
    detail["blocks"] = HIER_MEMORY.registry
    return jsonify(detail)

@app.route("/api/memory/promote")
def api_promote():
    if HIER_MEMORY is None: return jsonify({"error":"no manager"}), 404
    bid = request.args.get("id")
    target = request.args.get("target")  # facultatif
    # retrouver bloc (id tronqué affiché = 8 chars)
    real_id = None
    for full in HIER_MEMORY.registry.keys():
        if full.startswith(bid):
            real_id = full; break
    if not real_id: return jsonify({"error":"block not found"}), 404
    from core.memory_block import MemoryBlock
    mb = MemoryBlock(size_mb=HIER_MEMORY.registry[real_id]['size_mb'], gpu_id=0)
    mb.id = real_id
    if target:
        HIER_MEMORY.migrate(mb, target)
    else:
        # promotion simple : use promote_policy after touch
        HIER_MEMORY.touch(mb)
        HIER_MEMORY.promote_policy(mb)
    return jsonify({"ok":True, "tier": HIER_MEMORY.get_tier(real_id)})

@app.route("/api/memory/demote")
def api_demote():
    if HIER_MEMORY is None: return jsonify({"error":"no manager"}), 404
    bid = request.args.get("id")
    real_id = None
    for full in HIER_MEMORY.registry.keys():
        if full.startswith(bid):
            real_id = full; break
    if not real_id: return jsonify({"error":"block not found"}), 404
    order = ["L1","L2","L3","L4","L5","L6"]
    tier = HIER_MEMORY.get_tier(real_id)
    idx = order.index(tier)
    if idx < len(order)-1:
        from core.memory_block import MemoryBlock
        mb = MemoryBlock(size_mb=HIER_MEMORY.registry[real_id]['size_mb'], gpu_id=0)
        mb.id = real_id
        HIER_MEMORY.migrate(mb, order[idx+1])
    return jsonify({"ok":True, "tier": HIER_MEMORY.get_tier(real_id)})

@app.route("/switch")
def switch():
    mode = request.args.get("mode", "cli")
    subprocess.Popen(["python3", "launcher.py", "--mode", mode])
    return f"<p>Lancement de l’interface {mode}…</p><meta http-equiv='refresh' content='2;url=/' />"

if socketio:
    @socketio.on('subscribe_memory')
    def handle_subscribe_mem(msg):  # pragma: no cover
        if HIER_MEMORY:
            emit('memory', HIER_MEMORY.summary())

def push_memory_periodic():  # pragma: no cover - thread
    if not socketio:
        return
    import time
    while True:
        if HIER_MEMORY:
            try:
                socketio.emit('memory', HIER_MEMORY.summary())
            except Exception:
                pass
        time.sleep(3)


@app.route("/node")
def api_mobile_node():
    import os
    path = os.path.join(os.path.dirname(__file__), "templates", "mobile_edge_node.html")
    with open(path, "r") as f:
        return f.read()

def launch_in_thread(port=8500, host="0.0.0.0"):
    """Lance le dashboard dans un thread (werkzeug make_server : thread-safe, sans signaux).

    Utilisé par `vramancer serve` pour exposer le dashboard cluster en arrière-plan.
    """
    import threading
    from werkzeug.serving import make_server
    srv = make_server(host, port, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def launch(port=None, host="0.0.0.0"):
    import threading
    port = int(port or os.environ.get("VRM_DASHBOARD_PORT", "8500"))
    _debug = (os.environ.get("VRM_DEBUG", "0") == "1" and os.environ.get("VRM_PRODUCTION", "0") != "1")
    print(f"  VRAMancer dashboard → http://localhost:{port}  (Ctrl-C pour arrêter)", flush=True)
    if socketio:
        threading.Thread(target=push_memory_periodic, daemon=True).start()
        socketio.run(app, debug=_debug, use_reloader=False, host=host, port=port)
    else:
        # Fallback Flask pur (sans temps réel SocketIO)
        app.run(debug=_debug, host=host, port=port)
