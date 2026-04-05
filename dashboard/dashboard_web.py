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

def launch():
    import threading
    if socketio:
        threading.Thread(target=push_memory_periodic, daemon=True).start()
        socketio.run(app, debug=(os.environ.get("VRM_DEBUG", "0") == "1" and os.environ.get("VRM_PRODUCTION", "0") != "1"), use_reloader=False, host="0.0.0.0", port=8500)
    else:
        # Fallback Flask pur
        app.run(debug=(os.environ.get("VRM_DEBUG", "0") == "1" and os.environ.get("VRM_PRODUCTION", "0") != "1"), host="0.0.0.0", port=8500)
