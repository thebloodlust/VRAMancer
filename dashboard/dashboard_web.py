if os.environ.get('VRM_DASHBOARD_MINIMAL','0') == '0':
    # Auto-activation si dépendances lourdes manquantes détectées
    try:
        import importlib
        for mod in ('torch','transformers','numpy'):
            if importlib.util.find_spec(mod) is None:
                os.environ['VRM_DASHBOARD_MINIMAL'] = '1'
                break
    except Exception:
        os.environ['VRM_DASHBOARD_MINIMAL'] = '1'
# dashboard/dashboard_web.py

from flask import Flask, render_template_string, request, jsonify
try:
    from flask_socketio import SocketIO, emit  # type: ignore
except ImportError:  # fallback sans temps réel
    SocketIO = None  # type: ignore
    def emit(*a, **k):
        return None
import os, sys
try:
    from utils.gpu_utils import get_available_gpus
except Exception:
    def get_available_gpus():
        return []
import subprocess
from core.security import install_security

app = Flask(__name__)
if SocketIO:
    socketio = SocketIO(app, cors_allowed_origins="*")
else:
    socketio = None
install_security(app)

# Objet mémoire hiérarchique injecté dynamiquement par main (set via set_memory_manager)
HIER_MEMORY = None

def set_memory_manager(hm):
    global HIER_MEMORY
    HIER_MEMORY = hm

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VRAMancer Web</title>
    <style>
        body { background-color: #121212; color: #eee; font-family: Arial, sans-serif; padding: 20px; }
        .gpu { background: #1e1e1e; padding: 15px; margin-bottom: 10px; border-radius: 8px; }
        .bar { height: 20px; background: linear-gradient(90deg,#00BFFF,#007ACC); margin-top: 5px; border-radius: 4px; }
        select, button { margin-top: 10px; padding: 5px; font-size: 1em; }
        table { border-collapse: collapse; margin-top: 10px; }
        th, td { border: 1px solid #333; padding: 6px 10px; }
        a { color: #00BFFF; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .muted { color: #666; }
    </style>
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js" crossorigin="anonymous"></script>
</head>
<body>
    <h1>🧠 VRAMancer Web Dashboard</h1>

    <form method="get" action="/switch">
        <label for="mode">🎛️ Changer d’interface :</label>
        <select name="mode">
            <option value="cli">CLI</option>
            <option value="tk">Tkinter</option>
            <option value="qt">Qt</option>
        </select>
        <button type="submit">Lancer</button>
    </form>

    <h2>Hiérarchie Mémoire (temps réel)</h2>
    <table>
        <thead><tr><th>Tier</th><th>Count</th></tr></thead>
        <tbody id="tiers-body">
            {% if memory %}
            {% for tier, count in memory.tiers.items() %}
            <tr><td>{{ tier }}</td><td>{{ count }}</td></tr>
            {% endfor %}
            {% else %}
            <tr><td colspan="2" class="muted">(Pas de données)</td></tr>
            {% endif %}
        </tbody>
    </table>
    <h3>Blocs</h3>
    <table>
        <thead><tr><th>ID</th><th>Tier</th><th>Size(MB)</th><th>Access</th><th>Dernier accès</th><th>Actions</th></tr></thead>
        <tbody id="blocks-body">
            {% if memory %}
            {% for b in memory.blocks %}
            <tr>
                <td>{{ b.id }}</td>
                <td>{{ b.tier }}</td>
                <td>{{ b.size }}</td>
                <td>{{ b.access }}</td>
                <td>{{ b.last_access or '' }}</td>
                <td>
                    <a href="/api/memory/promote?id={{ b.id }}">⬆</a>
                    <a href="/api/memory/demote?id={{ b.id }}">⬇</a>
                </td>
            </tr>
            {% endfor %}
            {% else %}
            <tr><td colspan="6" class="muted">(Pas de blocs)</td></tr>
            {% endif %}
        </tbody>
    </table>

    <h2>GPUs</h2>
    {% for gpu in gpus %}
    <div class="gpu">
        <h3>🎮 {{ gpu.name }}</h3>
        <p>💾 VRAM: {{ gpu.used_vram_mb }} / {{ gpu.total_vram_mb }} MB</p>
        <div class="bar" style="width: {{ gpu.used_vram_mb / gpu.total_vram_mb * 100 }}%;"></div>
        <p>📡 Statut: {{ "✅ Disponible" if gpu.is_available else "❌ Indisponible" }}</p>
    </div>
    {% endfor %}

    <h2>📋 Tâches (Scheduler)</h2>
    <table>
        <thead><tr><th>Ressource</th><th>Priorité</th><th>ID tâche</th><th>Annuler</th></tr></thead>
        <tbody id="tasks-body"><tr><td colspan="4" class="muted">(Chargement)</td></tr></tbody>
    </table>
    <button onclick="submitNoop()">Ajouter noop</button>
    <button onclick="submitWarmup()">Warmup</button>
    <button onclick="fetchHistory()">History</button>
    <pre id="task-history" style="background:#1e1e1e;padding:8px;max-height:160px;overflow:auto;"></pre>
    <h3>Stats (p95/p99)</h3>
    <pre id="task-metrics" style="background:#1e1e1e;padding:8px;max-height:120px;overflow:auto;"></pre>

    <script>
    const socket = io();
    socket.on('connect', () => {
        socket.emit('subscribe_memory', {});
    });
    socket.on('memory', (data) => {
        if(!data) return;
        if(data.tiers){
            const tb = document.getElementById('tiers-body');
            tb.innerHTML = Object.entries(data.tiers).map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('');
        }
        if(data.blocks){
            const bb = document.getElementById('blocks-body');
            const rows = Object.entries(data.blocks).map(([bid, meta]) => {
                const shortId = bid.substring(0,8);
                const tier = meta.tier || meta.tier === '' ? meta.tier : meta.tier;
                const size = meta.size_mb || '';
                const access = meta.access || '';
                const last = meta.last_access || '';
                return `<tr><td>${shortId}</td><td>${tier}</td><td>${size}</td><td>${access}</td><td>${last}</td>`+
                       `<td><a href="/api/memory/promote?id=${shortId}">⬆</a> <a href="/api/memory/demote?id=${shortId}">⬇</a></td></tr>`;
            });
            bb.innerHTML = rows.join('');
        }
    });

        async function refreshTasks(){
            try {
                const r = await fetch('/api/tasks/status');
                if(!r.ok) return; const js = await r.json();
                const tb = document.getElementById('tasks-body');
                        const rows = Object.entries(js.active||{}).map(([res, info])=>{
                                const pri = info.priority || info;
                                const id = info.id || info.task_id || 'n/a';
                                return `<tr><td>${res}</td><td>${pri}</td><td>${id}</td><td><button onclick=cancelTask('${id}')>X</button></td></tr>`;
                        });
                        tb.innerHTML = rows.length? rows.join('') : '<tr><td colspan="4" class="muted">(Aucune tâche)</td></tr>';
            } catch(e) {}
            setTimeout(refreshTasks, 3000);
        }
        refreshTasks();
        async function submitNoop(){ await fetch('/api/tasks/submit', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({kind:'noop'})}); }
        async function submitWarmup(){ await fetch('/api/tasks/submit', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({kind:'warmup', priority:'HIGH'})}); }
        async function fetchHistory(){ const r = await fetch('/api/tasks/history'); if(r.ok){ const h = await r.json(); document.getElementById('task-history').textContent = JSON.stringify(h.slice(-10), null, 2);} }
        async function cancelTask(id){ await fetch('/api/tasks/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})}); }
        async function refreshMetrics(){try{const r=await fetch('/api/tasks/metrics');if(r.ok){const m=await r.json();document.getElementById('task-metrics').textContent=JSON.stringify(m,null,2);}}catch(e){} setTimeout(refreshMetrics,5000);} refreshMetrics();
    </script>
</body>
</html>
"""

@app.route("/")
def dashboard():
    gpus = get_available_gpus()
    memory = None
    if HIER_MEMORY is not None:
        summary = HIER_MEMORY.summary()
        # enrichir avec blocs
        blocks = []
        for bid, meta in HIER_MEMORY.registry.items():
            blocks.append({
                "id": bid[:8],
                "tier": meta["tier"],
                "size": meta.get("size_mb"),
                "access": meta.get("access"),
                "last_access": meta.get("last_access")
            })
        memory = type("_Mem", (), {"tiers": summary["tiers"], "blocks": blocks})
    return render_template_string(TEMPLATE, gpus=gpus, memory=memory)

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

def launch():
    import threading
    if socketio:
        threading.Thread(target=push_memory_periodic, daemon=True).start()
        socketio.run(app, debug=False, host="0.0.0.0", port=5000)
    else:
        # Fallback Flask pur
        app.run(host="0.0.0.0", port=5000)
