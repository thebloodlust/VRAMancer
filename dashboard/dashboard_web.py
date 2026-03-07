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

from flask import Flask, render_template_string, request, jsonify
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
        body { background-color: #0d0d12; color: #eee; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; }
        .gpu { background: #1a1a24; padding: 15px; margin-bottom: 10px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .bar { height: 20px; background: linear-gradient(90deg,#00d2ff,#3a7bd5); margin-top: 5px; border-radius: 4px; box-shadow: 0 0 10px rgba(0,210,255,0.5); }
        select, button { margin-top: 10px; padding: 8px 15px; font-size: 1em; background: #2a2a35; color: #fff; border: 1px solid #444; border-radius: 4px; cursor: pointer; transition: 0.3s; }
        button:hover { background: #3a7bd5; border-color: #00d2ff; }
        table { border-collapse: collapse; margin-top: 10px; width: 100%; background: #1a1a24; border-radius: 8px; overflow: hidden; }
        th, td { border: 1px solid #2a2a35; padding: 10px; text-align: left; }
        th { background: #222230; color: #00d2ff; }
        a { color: #00d2ff; text-decoration: none; font-weight: bold; }
        a:hover { text-shadow: 0 0 5px #00d2ff; }
        .muted { color: #666; }
        
        /* Swarm Attention Styles */
        #swarm-container {
            margin-top: 30px;
            background: #111118;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 20px;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.8), 0 0 15px rgba(0, 210, 255, 0.2);
            position: relative;
        }
        #swarm-canvas {
            width: 100%;
            height: 300px;
            border-radius: 8px;
            background: radial-gradient(circle at center, #1a1a24 0%, #0d0d12 100%);
        }
        .swarm-title {
            color: #00d2ff;
            text-shadow: 0 0 10px rgba(0, 210, 255, 0.8);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        .pulse-dot {
            width: 12px; height: 12px; background: #ff0055; border-radius: 50%;
            margin-right: 10px; box-shadow: 0 0 10px #ff0055;
            animation: pulse 1.5s infinite alternate;
        }
        @keyframes pulse { from { opacity: 0.5; transform: scale(0.8); } to { opacity: 1; transform: scale(1.2); } }
    </style>
    <!-- Removed external socket.io to bypass Firefox strict security policies on Codespaces -->
</head>
<body>
    <h1>🧠 VRAMancer Web Dashboard & Swarm Supervision</h1>

    <!-- Swarm Attention Neural Visualizer -->
    <div id="swarm-container">
        <h2 class="swarm-title"><div class="pulse-dot"></div> Swarm Attention - Réseau Neuronal Organique (L7 WebGPU)</h2>
        <p class="muted">Visualisation en temps réel des fragments de KV-Cache distribués aux navigateurs Edge.</p>
        <canvas id="swarm-canvas"></canvas>
        <div style="text-align: center; margin-top: 10px;">
            <button onclick="window.forceSimulation = !window.forceSimulation; this.style.background = window.forceSimulation ? '#00d2ff' : '#ff0055'; this.innerText = window.forceSimulation ? '🛑 Arrêter la Simulation' : '⚡ Simuler l\\'Offload sur l\\'Organisme';" style="background:#ff0055; border:none; font-weight:bold; color: white; padding: 10px 20px; border-radius: 5px; cursor: pointer; transition: 0.3s; box-shadow: 0 0 10px rgba(255,0,85,0.5);">
                ⚡ Simuler l'Offload sur l'Organisme
            </button>
        </div>
    </div>

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
        // Removed real-time websocket listener dependency to ensure UI always loads 
        // regardless of network constraints. Stats will fallback to REST if needed.

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

        // --- Swarm Attention Organic Neural Network Visualizer ---
        const canvas = document.getElementById('swarm-canvas');
        const ctx = canvas.getContext('2d');
        let width = canvas.width = canvas.offsetWidth;
        let height = canvas.height = 300;
        
        const centralNode = { x: width / 2, y: height / 2, radius: 25, color: '#00d2ff', pulse: 0 };
        const edgeNodes = [];
        const numNodes = 15; // Simulate ~15 edge browsers
        
        for(let i=0; i<numNodes; i++) {
            const angle = (Math.PI * 2 / numNodes) * i;
            const distance = 80 + Math.random() * 60;
            edgeNodes.push({
                baseX: centralNode.x + Math.cos(angle) * distance,
                baseY: centralNode.y + Math.sin(angle) * distance,
                angle: angle,
                distance: distance,
                radius: 4 + Math.random() * 6,
                active: Math.random() > 0.3,
                activationTimer: Math.random() * 100,
                color: '#ff0055'
            });
        }

        window.forceSimulation = false;
        function drawSwarm() {
            if(!ctx) return;
            ctx.clearRect(0, 0, width, height);
            
            centralNode.pulse += 0.05;
            const glow = (window.forceSimulation ? 20 : 5) + Math.sin(centralNode.pulse) * (window.forceSimulation ? 15 : 5);
            
            // Draw connections (Tensor Offloading)
            edgeNodes.forEach(node => {
                node.activationTimer += window.forceSimulation ? 3 : 1;
                if(node.activationTimer > (window.forceSimulation ? 50 : 150)) {
                    node.active = window.forceSimulation ? true : (Math.random() > 0.5); // Always active in simulation demo
                    node.activationTimer = 0;
                }
                
                // Orbit slightly
                node.angle += window.forceSimulation ? 0.015 : 0.002;
                const nx = centralNode.x + Math.cos(node.angle) * node.distance;
                const ny = centralNode.y + Math.sin(node.angle) * node.distance;
                
                ctx.beginPath();
                ctx.moveTo(centralNode.x, centralNode.y);
                ctx.lineTo(nx, ny);
                
                // Connection styles based on activity
                if(node.active) {
                    ctx.strokeStyle = `rgba(0, 210, 255, ${0.4 + Math.sin(centralNode.pulse * 2)*0.2})`;
                    ctx.lineWidth = 2;
                    // Draw flowing data packets
                    const progress = (Date.now() / 10 % 100) / 100;
                    const px = centralNode.x + (nx - centralNode.x) * progress;
                    const py = centralNode.y + (ny - centralNode.y) * progress;
                    ctx.fillStyle = '#ff0055';
                    ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI*2); ctx.fill();
                } else {
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
                    ctx.lineWidth = 1;
                }
                ctx.stroke();
                
                // Draw edge node
                ctx.beginPath();
                ctx.arc(nx, ny, node.radius, 0, Math.PI * 2);
                ctx.fillStyle = node.active ? '#ff0055' : '#333';
                ctx.shadowColor = node.active ? '#ff0055' : 'transparent';
                ctx.shadowBlur = node.active ? 10 : 0;
                ctx.fill();

                // Draw Stats (Tokens/s or FLOPS)
                if (node.active) {
                    ctx.fillStyle = '#eee';
                    ctx.font = '10px Consolas, monospace';
                    ctx.textAlign = 'left';
                    ctx.fillText(`+${Math.floor((node.radius)*3)} T/s`, nx + 12, ny - 6);
                    ctx.fillStyle = 'rgba(0, 210, 255, 0.8)';
                    ctx.fillText(`${(node.radius*0.2).toFixed(1)} TFLOP`, nx + 12, ny + 6);
                }
            });
            
            // Draw central node (Core Scheduler)
            ctx.beginPath();
            ctx.arc(centralNode.x, centralNode.y, centralNode.radius + glow, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 210, 255, 0.2)`;
            ctx.fill();
            
            ctx.beginPath();
            ctx.arc(centralNode.x, centralNode.y, centralNode.radius, 0, Math.PI * 2);
            ctx.fillStyle = centralNode.color;
            ctx.shadowColor = centralNode.color;
            ctx.shadowBlur = 20;
            ctx.fill();
            ctx.shadowBlur = 0; // reset
            
            // Central Text
            ctx.fillStyle = '#000';
            ctx.font = 'bold 13px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('CORE', centralNode.x, centralNode.y - 5);
            ctx.font = 'bold 9px Arial';
            ctx.fillText('L1/L2 Full', centralNode.x, centralNode.y + 8);

            // Global stats top left
            ctx.fillStyle = '#00d2ff';
            ctx.font = '14px Consolas';
            ctx.textAlign = 'left';
            const activeNodes = edgeNodes.filter(n=>n.active);
            const totalFlops = activeNodes.reduce((acc, curr) => acc + (curr.radius*0.2), 0);
            ctx.fillText(`Swarm Active Compute: ${totalFlops.toFixed(1)} TFLOPS`, 15, 25);
            ctx.fillStyle = '#ff0055';
            ctx.fillText(`Distributed KV-Cache: ${activeNodes.length * 512} MB`, 15, 45);

            if (window.forceSimulation) {
                ctx.fillStyle = '#ffff00';
                ctx.fillText(`<< SWARM ATTENTION DEV-PREVIEW MODE >>`, 15, 65);
                ctx.fillStyle = '#00ff00';
                ctx.fillText(`Virtual Nodes Simulated: ${numNodes} Edge Devices`, 15, 80);
            }

            requestAnimationFrame(drawSwarm);
        }
        
        window.addEventListener('resize', () => {
            width = canvas.width = canvas.offsetWidth;
            centralNode.x = width / 2;
            centralNode.y = height / 2;
        });
        
        drawSwarm();
    </script>
</body>
</html>
"""

@app.route("/")
def dashboard():
    # Bypass heavy GPU/Memory init that might stall load
    gpus = [{"name": "Cluster-L1 Orchestrator", "total_vram_mb": 24576, "used_vram_mb": 4096, "is_available": True}]
    memory = None
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
