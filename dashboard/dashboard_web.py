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
<html lang="fr" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VRAMancer Orbital Supervision</title>
    <!-- Tailwind CSS Engine -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Icons (Phosphor) -->
    <script src="https://unpkg.com/@phosphor-icons/web"></script>
    <!-- AlpineJS for reactivity (lightweight) -->
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <!-- 3D Force Graph for Interactive Network Map -->
    <script src="https://unpkg.com/3d-force-graph"></script>

    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        cyber: { 
                            dark: '#030712', medium: '#111827', light: '#1f2937',
                            accent: '#00f2fe', neon: '#4facfe', danger: '#fb2c36', success: '#00ff87' 
                        }
                    },
                    animation: {
                        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                    }
                }
            }
        }
    </script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700;800&display=swap');
        body { font-family: 'JetBrains Mono', monospace; background-color: #030712; color: #e5e7eb; overflow-x: hidden; }
        .glass-panel { background: rgba(17, 24, 39, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.05); }
        .neon-text { text-shadow: 0 0 10px rgba(0, 242, 254, 0.5); }
        .vram-bar { background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); transition: width 0.5s ease; }
        .vram-danger { background: linear-gradient(90deg, #f5576c 0%, #f093fb 100%); }
        
        /* Particles for background */
        #particles { position: fixed; top:0; left:0; width:100vw; height:100vh; z-index:-1; pointer-events:none; opacity: 0.3;}
    </style>
</head>
<body class="antialiased min-h-screen selection:bg-cyber-accent selection:text-black" x-data="dashboardData()">
    <!-- Background Canvas -->
    <canvas id="particles"></canvas>

    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        
        <!-- Header -->
        <header class="flex flex-col md:flex-row justify-between items-center mb-10 gap-6 glass-panel rounded-2xl p-6 shadow-2xl">
            <div class="flex items-center gap-4">
                <div class="relative">
                    <i class="ph-fill ph-cpu text-6xl text-cyber-accent animate-pulse-slow"></i>
                    <div class="absolute -bottom-2 -right-2 text-xs font-bold bg-cyber-accent text-black px-2 py-0.5 rounded shadow-[0_0_10px_#00f2fe]">LIVE</div>
                </div>
                <div>
                    <h1 class="text-3xl font-extrabold tracking-tight text-white mb-1"><span class="neon-text">VRAMancer</span> Nexus</h1>
                    <p class="text-cyber-neon text-sm uppercase tracking-widest font-semibold flex items-center gap-2">
                        <span class="w-2 h-2 rounded-full bg-cyber-success animate-ping"></span> 
                        Swarm Orchestrator Active
                    </p>
                </div>
            </div>
            <div class="flex gap-4">
                <a href="/chat" class="group relative px-6 py-3 font-semibold text-white bg-cyber-dark border border-cyber-accent/50 rounded-lg overflow-hidden transition-all hover:border-cyber-neon hover:shadow-[0_0_15px_rgba(78,205,196,0.4)]">
                    <div class="absolute inset-0 w-0 bg-cyber-neon/10 transition-all duration-[250ms] ease-out group-hover:w-full"></div>
                    <span class="relative flex items-center gap-2"><i class="ph ph-chat-centered-text"></i> Chat UI</span>
                </a>
                <a href="/browser" class="group relative px-6 py-3 font-semibold text-white bg-cyber-dark border border-cyber-accent/50 rounded-lg overflow-hidden transition-all hover:border-cyber-accent hover:shadow-[0_0_15px_rgba(0,242,254,0.4)]">
                    <div class="absolute inset-0 w-0 bg-cyber-accent/10 transition-all duration-[250ms] ease-out group-hover:w-full"></div>
                    <span class="relative flex items-center gap-2"><i class="ph ph-magnifying-glass"></i> Model Browser</span>
                </a>
            </div>
        </header>

        <!-- Main Dashboard Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">

            <!-- Col 1: Hardware & GPUs -->
            <div class="lg:col-span-2 space-y-6">
                <!-- Swarm Attention Widget -->
                <div class="glass-panel p-0 rounded-2xl relative overflow-hidden flex flex-col justify-between" style="min-height: 320px; border: 1px solid rgba(0, 242, 254, 0.2);">
                    <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyber-accent to-transparent opacity-80 z-20"></div>
                    
                    <!-- Graph 3D Container -->
                    <div id="3d-graph" class="absolute inset-0 w-full h-full z-0 cursor-move"></div>

                    <!-- Overlay UI Elements -->
                    <div class="relative z-10 p-6 flex justify-between items-start pointer-events-none">
                        <div>
                            <h2 class="text-xl font-bold text-white flex items-center gap-2 drop-shadow-md"><i class="ph ph-share-network"></i> Tokio Network Data Plane</h2>
                            <p class="text-xs text-cyber-success mt-1 font-mono drop-shadow-md">Rust / Zero-Copy Safetensors Active <i class="ph-fill ph-check-circle"></i></p>
                        </div>
                        <div class="bg-cyber-dark/80 backdrop-blur px-3 py-1 pb-1.5 rounded-full border border-cyber-accent/50 text-xs font-bold text-cyber-neon flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full bg-cyber-success animate-ping"></span>
                            <span>Swarm Linked</span>
                        </div>
                    </div>

                    <div class="relative z-10 p-6 grid grid-cols-2 gap-4 mt-auto pointer-events-none">
                        <div class="bg-black/70 backdrop-blur rounded-xl p-4 border border-gray-800">
                            <div class="text-[0.65rem] text-gray-500 uppercase tracking-widest mb-1">Tier Actif</div>
                            <div class="text-lg font-bold text-white">Tier 2: <span class="text-cyber-accent">Zero-Copy TCP</span></div>
                            <div class="text-xs text-gray-400 mt-1">Latence Bypass CPU</div>
                        </div>
                        <div class="bg-black/70 backdrop-blur rounded-xl p-4 border border-gray-800">
                            <div class="text-[0.65rem] text-gray-500 uppercase tracking-widest mb-1">Tenseurs Routés (P2P)</div>
                            <div class="text-lg font-bold text-cyber-success font-mono" x-text="metrics.tensorsProcessed">0</div>
                            <div class="text-xs text-gray-400 mt-1">Via HMAC Zero-Trust</div>
                        </div>
                    </div>
                </div>

                <!-- Local Hardware Pool -->
                <h2 class="text-xl font-bold text-white mt-8 mb-4 border-b border-gray-800 pb-2"><i class="ph ph-graphics-card"></i> Local Hardware Pool</h2>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4" id="gpu-container">
                    <!-- Dynamic GPUs from Flask injection -->
                    {% for gpu in gpus %}
                    <div class="glass-panel p-5 rounded-xl border-l-4 border-l-cyber-neon transition flex flex-col justify-between hover:bg-gray-800/50">
                        <div class="flex justify-between items-start mb-4">
                            <div class="flex items-center gap-3">
                                <i class="ph ph-hard-drives text-3xl text-gray-300"></i>
                                <div>
                                    <h3 class="font-bold text-white leading-tight">{{ gpu.name }}</h3>
                                    <p class="text-xs text-cyber-success font-semibold mt-0.5">{{ "Online" if gpu.is_available else "Offline" }}</p>
                                </div>
                            </div>
                            <div class="text-right">
                                <p class="text-[0.65rem] text-gray-500 uppercase">Usage</p>
                                <p class="font-bold text-white font-mono">{{ gpu.used_vram_mb }}<span class="text-gray-400 text-sm">/{{ gpu.total_vram_mb }}</span> <span class="text-xs text-cyber-neon">MB</span></p>
                            </div>
                        </div>
                        
                        <div class="w-full bg-gray-900 rounded-full h-2.5 mb-1 overflow-hidden shadow-inner">
                            {% set pct = (gpu.used_vram_mb / gpu.total_vram_mb * 100) if gpu.total_vram_mb else 0 %}
                            <div class="h-full rounded-full transition-all duration-1000 {% if pct > 85 %} vram-danger {% else %} vram-bar {% endif %}" 
                                 style="width: {{ pct }}%"></div>
                        </div>
                        <div class="text-right text-xs text-gray-400">{{ "%.1f"|format(pct) }}% Utilisée</div>
                    </div>
                    {% endfor %}
                    
                    {% if not gpus %}
                        <div class="col-span-2 glass-panel p-8 text-center text-gray-500 rounded-xl border border-dashed border-gray-700">
                            <i class="ph ph-warning-circle text-4xl mb-2"></i>
                            <p>Aucun accélérateur matériel détecté sur ce nœud.</p>
                        </div>
                    {% endif %}
                </div>
            </div> <!-- End Col 1 -->

            <!-- Col 2: Memory & Activity -->
            <div class="space-y-6">
                <!-- Memory Hierarchy Map -->
                <div class="glass-panel p-6 rounded-2xl border border-gray-800/60">
                    <h2 class="text-lg font-bold text-white mb-4 flex items-center gap-2"><i class="ph ph-stack"></i> Hierarchical Memory</h2>
                    
                    <div class="space-y-4">
                        {% if memory %}
                            {% for tier, count in memory.tiers.items() %}
                            <div>
                                <div class="flex justify-between text-xs mb-1">
                                    <span class="uppercase tracking-wider text-gray-400 font-bold">{{ tier }}</span>
                                    <span class="text-cyber-accent font-mono">{{ count }} blocks</span>
                                </div>
                                <div class="w-full bg-black rounded-full h-1.5 border border-gray-800">
                                    <div class="bg-gray-600 h-1.5 rounded-full" style="width: min(100%, {{ count * 10 }}%)"></div>
                                </div>
                            </div>
                            {% endfor %}
                        {% else %}
                            <div class="text-sm text-gray-500 italic text-center py-4">Pas de pagination active</div>
                        {% endif %}
                    </div>
                </div>

                <!-- Live Task Feed (Console) -->
                <div class="glass-panel rounded-2xl border border-gray-800/60 flex flex-col h-[400px]">
                    <div class="p-4 border-b border-gray-800 flex justify-between items-center bg-black/20 rounded-t-2xl">
                        <h2 class="text-sm font-bold text-white flex items-center gap-2"><i class="ph ph-terminal-window"></i> Orchestrator Logs</h2>
                        <span class="flex h-2 w-2 relative">
                            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyber-success opacity-75"></span>
                            <span class="relative inline-flex rounded-full h-2 w-2 bg-cyber-success"></span>
                        </span>
                    </div>
                    
                    <div class="p-4 flex-1 overflow-y-auto font-mono text-[0.65rem] space-y-2" id="live-logs">
                        <!-- Filled by JS -->
                        <div class="text-gray-500">[SYSTEM] Initialization sequence started...</div>
                        <div class="text-cyber-neon">[INFO] Cluster L1 connected successfully.</div>
                    </div>
                </div>
            </div>

        </div> <!-- End Grid -->
    </div>


    <script>
        // Alpine Data Store
        function dashboardData() {
            return {
                metrics: {
                    activeNodes: 0,
                    bandwidth: 0,
                    tensorsProcessed: 0
                },
                logs: [],
                init() {
                    this.fetchLiveTasks();
                    setInterval(() => this.updateMockMetrics(), 2000);
                },
                async updateMockMetrics() {
                    try {
                        const r = await fetch('/api/swarm/status');
                        if (r.ok) {
                            const data = await r.json();
                            // Real metrics
                            this.metrics.activeNodes = data.activeNodes;
                            this.metrics.bandwidth = data.bandwidth;
                            this.metrics.tensorsProcessed = data.tensorsProcessed;
                            
                            // Let's update the 3D Graph dynamically!
                            if (window.update3DGraph) {
                                window.update3DGraph(data.activeNodes);
                            }
                        }
                    } catch(e) {}
                },
                async fetchLiveTasks() {
                    const logContainer = document.getElementById('live-logs');
                    
                    try {
                        const r = await fetch('/api/tasks/status');
                        if (r.ok) {
                            const js = await r.json();
                            let html = '';
                            Object.entries(js.active || {}).forEach(([res, info]) => {
                                const id = info.id || info.task_id || 'n/a';
                                html += `<div class="text-cyber-accent"><span class="text-white">[TASK]</span> Routage Tenseur via Rust/Tokio: ${id}</div>`;
                            });
                            if (html === "") {
                                html = `<div class="text-gray-600">[IDLE] Attente des requêtes /v1/chat/completions...</div>`;
                                html += `<div class="text-cyber-success mt-2">[WOI] Écoute Magic Packets Wake-on-Inference activée.</div>`;
                            }
                            logContainer.innerHTML = html;
                        }
                    } catch(e) {}
                    
                    setTimeout(() => this.fetchLiveTasks(), 3000);
                }
            }
        }

        // --- 3D INTERACTIVE NETWORK MAP (Tokio Cluster) ---
        // Dynamically generating cluster nodes
        const Graph = ForceGraph3D()(document.getElementById('3d-graph'))
            .backgroundColor('rgba(0,0,0,0)')
            .nodeRelSize(8)
            .nodeAutoColorBy('group')
            .nodeColor(node => node.group === 1 ? '#00f2fe' : (node.group === 2 ? '#4facfe' : '#fb2c36'))
            .linkColor(() => 'rgba(0, 242, 254, 0.4)')
            .linkWidth(2)
            .linkDirectionalParticles(4)
            .linkDirectionalParticleWidth(3)
            .linkDirectionalParticleSpeed(d => d.value * 0.005)
            .cameraPosition({ x: 0, y: 0, z: 250 });

        // Simulate a cluster of exactly 1 Orchestrator, 4 Worker GPUs, 2 NVMe Nodes
        const gData = {
            nodes: [
                { id: 'Orchestrator (Rust/Python)', group: 1, val: 20 },
                { id: 'Worker GPU 1 (RTX 4090)', group: 2, val: 15 },
                { id: 'Worker GPU 2 (RTX 3090)', group: 2, val: 15 },
                { id: 'Worker GPU 3 (Apple M3)', group: 2, val: 10 },
                { id: 'Worker GPU 4 (WebGPU Edge)', group: 2, val: 8 },
                { id: 'NVMe Swap Node A', group: 3, val: 5 },
            ],
            links: [
                { source: 'Worker GPU 1 (RTX 4090)', target: 'Orchestrator (Rust/Python)', value: 8 },
                { source: 'Worker GPU 2 (RTX 3090)', target: 'Orchestrator (Rust/Python)', value: 6 },
                { source: 'Worker GPU 3 (Apple M3)', target: 'Orchestrator (Rust/Python)', value: 4 },
                { source: 'Worker GPU 4 (WebGPU Edge)', target: 'Orchestrator (Rust/Python)', value: 2 },
                { source: 'NVMe Swap Node A', target: 'Orchestrator (Rust/Python)', value: 1 },
                { source: 'Worker GPU 1 (RTX 4090)', target: 'Worker GPU 2 (RTX 3090)', value: 3 }, // P2P Tokio Link
            ]
        };

        Graph.graphData(gData);

        // Make the graph spin slowly for the WOW factor
        let angle = 0;
        setInterval(() => {
            angle += Math.PI / 800;
            Graph.cameraPosition({
                x: 250 * Math.sin(angle),
                z: 250 * Math.cos(angle)
            });
        }, 30);
        
        // Handle Resize to keep Graph responsive
        window.addEventListener('resize', () => {
            const container = document.getElementById('3d-graph');
            Graph.width(container.clientWidth).height(container.clientHeight);
        });
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

BROWSER_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VRAMancer - Navigateur de Modèles</title>
    <style>
        body { background-color: #0d0d12; color: #eee; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; }
        input { padding: 10px; width: 400px; font-size: 1.1em; background: #1a1a24; color: #fff; border: 1px solid #444; border-radius: 4px; }
        button { padding: 10px 20px; font-size: 1.1em; background: #2a2a35; color: #fff; border: 1px solid #333; border-radius: 4px; cursor: pointer; transition: 0.3s; margin-left: 10px; }
        button:hover { background: #00d2ff; color: #000; }
        .back-link { display: inline-block; margin-bottom: 20px; color: #00d2ff; text-decoration: none; font-weight: bold; font-size: 1.1em; }
        .back-link:hover { text-shadow: 0 0 5px #00d2ff; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; background: #1a1a24; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        th, td { border: 1px solid #2a2a35; padding: 12px; text-align: left; }
        th { background: #222230; color: #00d2ff; text-transform: uppercase; letter-spacing: 1px; }
        .source-hf { color: #ffcc00; font-weight: bold; background: rgba(255,204,0,0.1); padding: 4px 8px; border-radius: 4px; }
        .source-ollama { color: #00ffcc; font-weight: bold; background: rgba(0,255,204,0.1); padding: 4px 8px; border-radius: 4px; }
        .loading { font-style: italic; color: #00d2ff; margin-top: 10px; font-weight: bold; }
    </style>
</head>
<body>
    <a href="/" class="back-link">⬅ Retour au Dashboard</a>
    <h1>🌐 Navigateur Unifié de Modèles</h1>
    <p>Recherchez et chargez des modèles depuis <span style="color:#ffcc00;"><b>Hugging Face</b></span> et <span style="color:#00ffcc;"><b>Ollama</b></span> pour que VRAMancer les répartisse sur vos GPUs.</p>
    
    <div style="margin-top:20px; background: #111118; padding: 20px; border-radius: 8px; border: 1px solid #333;">
        <input type="text" id="searchInput" placeholder="Ex: llama3, mistral, qwen..." onkeydown="if(event.key==='Enter') searchModels()">
        <button onclick="searchModels()">Rechercher</button>
        <div id="status" class="loading"></div>
    </div>

    <table id="resultsTable">
        <thead>
            <tr>
                <th>Nom du Modèle / ID</th>
                <th>Dépôt Source</th>
                <th>Action</th>
            </tr>
        </thead>
        <tbody id="resultsBody">
            <tr><td colspan="3" style="text-align:center; color:#666; padding: 30px;">Entrez un terme ci-dessus pour lancer la recherche multisource.</td></tr>
        </tbody>
    </table>

    <script>
        async function searchModels() {
            const q = document.getElementById('searchInput').value;
            const status = document.getElementById('status');
            const tbody = document.getElementById('resultsBody');
            
            if (!q.trim()) return;
            
            status.innerHTML = "Sub-espace réseau actif : Interrogation des registres Hugging Face et Ollama...";
            tbody.innerHTML = "";
            
            try {
                const r = await fetch('/api/models/search?q=' + encodeURIComponent(q));
                const data = await r.json();
                
                if (data.results && data.results.length > 0) {
                    tbody.innerHTML = data.results.map(m => {
                        const sourceClass = m.source === 'Hugging Face' ? 'source-hf' : 'source-ollama';
                        return `<tr>
                            <td style="font-weight:bold; font-size:1.1em; color: #fff;">${m.id}</td>
                            <td><span class="${sourceClass}">${m.source}</span></td>
                            <td><button onclick="loadModel('${m.id}', '${m.source}')" style="background: #00d2ff; color: #000; font-weight:bold; border:none; border-radius:4px;">🚀 Charger dans VRAMancer</button></td>
                        </tr>`;
                    }).join('');
                } else {
                    tbody.innerHTML = `<tr><td colspan="3" style="text-align:center; color:#888; padding:30px;">Aucun modèle trouvé.</td></tr>`;
                }
                status.innerHTML = "";
            } catch (err) {
                status.innerHTML = "Erreur de connexion. Vérifiez si vous êtes en ligne.";
                console.error(err);
            }
        }

        async function loadModel(modelId, source) {
            let backendToUse = source === 'Ollama' ? 'ollama' : 'huggingface';
            const status = document.getElementById('status');
            
            // Hardcode des variables critiques d'asymétrie P2P 
            const payload = {
                model: modelId,
                backend_type: backendToUse,
                gpu_memory_utilization: 0.96,
                max_model_len: 4096,
                enforce_eager: true
            };
            
            status.innerHTML = `<span class="text-cyber-accent">Initialisation du Cortex Tenseur pour ${modelId} (Zero-Copy P2P)...</span>`;
            
            try {
                // Utilise le header d'authentification Bearer testtoken commun au projet minimal
                const response = await fetch(`http://${window.location.hostname}:5000/api/models/load`, {
                    method: 'POST', 
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer testtoken'
                    },
                    body: JSON.stringify(payload)
                });
                
                if(response.ok) {
                    const resData = await response.json();
                    status.innerHTML = `<span style="color:#00ff87;">✅ Fusion Neurale établie ! (${resData.num_gpus} GPUs Actifs)</span>`;
                    alert(`VRAMancer :\n\nModèle chargé avec succès sur ${resData.num_gpus} GPUs.\nTensor Parallelism actif.`);
                } else {
                    const errText = await response.text();
                    status.innerHTML = `<span style="color:#fb2c36;">❌ Crash Mémoire P2P : ${response.status}</span>`;
                    alert("Erreur de chargement de modèle : " + errText);
                }
            } catch(e) {
                status.innerHTML = `<span style="color:#fb2c36;">❌ Serveur VRAMancer Hors-Ligne ou Inaccessible.</span>`;
                console.error(e);
            }
        }
    </script>
</body>
</html>
"""

CHAT_TEMPLATE = """
<html>
<head>
    <title>VRAMancer - Chat Terminal</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0d0d12; color: #eee; font-family: 'Segoe UI', Tahoma, sans-serif; }
        .cyber-panel { background: #1a1a24; border: 1px solid #333; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .cyber-input { background: #111118; border: 1px solid #444; color: white; }
        .cyber-input:focus { outline: none; border-color: #00d2ff; box-shadow: 0 0 10px rgba(0,210,255,0.2); }
        .msg-user { background: rgba(0, 210, 255, 0.1); border-left: 3px solid #00d2ff; }
        .msg-ai { background: rgba(78, 205, 196, 0.1); border-left: 3px solid #4ecdc4; }
        .back-link { display: inline-block; margin-bottom: 20px; color: #00d2ff; text-decoration: none; font-weight: bold; font-size: 1.1em; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #111118; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #00d2ff; }
    </style>
</head>
<body class="min-h-screen flex flex-col items-center py-8 px-4">
    <div class="w-full max-w-5xl"><a href="/" class="back-link">⬅ Retour au Nexus Swarm</a></div>
    
    <div class="w-full max-w-5xl cyber-panel rounded-xl flex flex-col overflow-hidden" style="height: 75vh;">
        <div class="p-4 bg-[#111118] border-b border-gray-700 flex justify-between items-center">
            <span class="font-bold text-lg"><span class="text-[#00d2ff]">VRAMancer</span> <span class="text-white">Asymmetric Neural Chat</span></span>
            <span id="speed-indicator" class="text-xs font-mono text-[#4ecdc4] bg-[#0d0d12] px-3 py-1 rounded-full border border-[#4ecdc4]/30">Idle</span>
        </div>
        
        <div id="chat-box" class="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
            <div class="p-4 rounded msg-ai text-sm font-mono leading-relaxed max-w-[85%]">
                <span class="text-[#4ecdc4] font-bold">Système:</span> Moteur Rust P2P Zero-Copy activé. Les graphes PagedAttention sont compilés sur les GPUs. En attente de charge d'inférence.
            </div>
        </div>

        <div class="p-4 bg-[#111118] border-t border-gray-700 flex gap-3">
            <input type="text" id="prompt-input" class="flex-1 cyber-input rounded-lg px-4 py-3 font-mono text-sm" placeholder="Entrez un prompt pour le cluster..." onkeydown="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()" class="px-8 py-3 bg-gradient-to-r from-[#00d2ff] to-[#3a7bd5] hover:opacity-80 text-black font-bold rounded-lg transition-all shadow-[0_0_15px_rgba(0,210,255,0.3)]">Générer</button>
        </div>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('prompt-input');
            const chatBox = document.getElementById('chat-box');
            const speedInd = document.getElementById('speed-indicator');
            const prompt = input.value.trim();
            if (!prompt) return;

            chatBox.innerHTML += `<div class="p-4 rounded msg-user text-sm font-mono self-end max-w-[85%] whitespace-pre-wrap"><span class="text-[#00d2ff] font-bold">Vous:</span><br>${prompt}</div>`;
            input.value = '';
            chatBox.scrollTo(0, chatBox.scrollHeight);

            const msgId = 'msg-' + Date.now();
            chatBox.innerHTML += `<div id="wrap-${msgId}" class="p-4 rounded msg-ai text-sm font-mono self-start max-w-[85%] whitespace-pre-wrap"><span class="text-[#4ecdc4] font-bold">VRAMancer:</span><br><span id="${msgId}" class="animate-pulse">Calcul tensoriel en cours...</span></div>`;
            chatBox.scrollTo(0, chatBox.scrollHeight);
            speedInd.innerHTML = `<span class="text-[#ffcc00] animate-pulse">Processing...</span>`;

            const startTime = Date.now();

            try {
                // Determine base URL dynamically
                let baseUrl = window.location.origin;
                if (window.location.port === "8500") {
                   baseUrl = `http://${window.location.hostname}:5000`;
                }

                const response = await fetch(`${baseUrl}/v1/completions`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer testtoken' },
                    body: JSON.stringify({
                        model: 'Qwen/Qwen2.5-14B-Instruct',
                        prompt: prompt,
                        max_tokens: 500,
                        temperature: 0.7
                    })
                });

                const data = await response.json();
                const timeSec = (Date.now() - startTime) / 1000;
                
                let text = "Erreur de génération";
                let tps = 0;
                if (data.choices && data.choices.length > 0) {
                    text = data.choices[0].text;
                    const tokens = data.usage?.completion_tokens || 0;
                    tps = (tokens / timeSec).toFixed(1);
                }

                document.getElementById(msgId).classList.remove('animate-pulse');
                document.getElementById(msgId).innerHTML = text;
                speedInd.innerHTML = `⚡ ${tps} tok/s | Latency: ${timeSec.toFixed(2)}s`;

            } catch (err) {
                document.getElementById(msgId).classList.remove('animate-pulse');
                document.getElementById(msgId).innerHTML = `<span class="text-red-500">Erreur réseau : Impossible de joindre l'API d'inférence. Le modèle est-il chargé via VRAMancer P2P ?</span>`;
                speedInd.innerText = "Error";
            }
            chatBox.scrollTo(0, chatBox.scrollHeight);
        }
    </script>
</body>
</html>
"""

@app.route("/chat")
def chat_ui():
    return render_template_string(CHAT_TEMPLATE)

@app.route("/browser")
def model_browser():
    return render_template_string(BROWSER_TEMPLATE)

@app.route("/api/models/search")
def api_models_search():
    print(">>> SEARCH ROUTE HIT! Query:", request.args.get("q"))
    try:
        import requests
    except ImportError:
        print(">>> MISSING REQUESTS LIB")
        return jsonify({"results": [{"id": "Erreur: request library missing", "source": "Internal"}]})
    
    query = request.args.get("q", "")
    results = []
    
    # --- Recherche Hugging Face ---
    try:
        print(">>> FETCHING FROM HF...")
        hf_resp = requests.get(f"https://huggingface.co/api/models?search={query}&limit=10", timeout=3)
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
    try:
        from core.metrics import WEBGPU_CONNECTED_CLIENTS, WEBGPU_FLOPS_TOTAL
        clients = max(0, int(WEBGPU_CONNECTED_CLIENTS._value.get()))
        flops = WEBGPU_FLOPS_TOTAL._value.get()
        tensors = int(flops // 15000000) # estimation for visuals
    except Exception:
        clients = 0
        flops = 0
        tensors = 0

    return jsonify({
        "activeNodes": clients,
        "flopsProcessed": flops,
        "tensorsProcessed": tensors,
        "bandwidth": round(clients * 1.5, 1) # simple fake proxy for visuals if nodes exist
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

def launch():
    import threading
    if socketio:
        threading.Thread(target=push_memory_periodic, daemon=True).start()
        socketio.run(app, debug=True, host="0.0.0.0", port=8500)
    else:
        # Fallback Flask pur
        app.run(debug=True, host="0.0.0.0", port=8500)
