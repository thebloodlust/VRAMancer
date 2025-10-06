"""
Extension de monitoring mobile pour clusters hétérogènes.
Support temps réel, visualisation compacte, gestion tactile.
"""
from flask import Flask, render_template_string, jsonify, request
import json
import time
import threading
from typing import Dict, List, Any
from core.orchestrator.heterogeneous_manager import get_heterogeneous_manager

app = Flask(__name__)

# Cache des métriques en temps réel
_metrics_cache = {
    "last_update": 0,
    "nodes": {},
    "cluster_health": "unknown"
}

def update_metrics_cache():
    """Met à jour le cache des métriques en arrière-plan."""
    while True:
        try:
            hetero_mgr = get_heterogeneous_manager()
            
            # Détection locale si pas de nœuds
            if not hetero_mgr.nodes:
                local_caps = hetero_mgr.detect_local_capabilities()
                hetero_mgr.register_node(local_caps)
            
            summary = hetero_mgr.get_cluster_summary()
            node_details = {}
            
            for hostname, caps in hetero_mgr.nodes.items():
                node_details[hostname] = {
                    "arch": caps.architecture,
                    "platform": caps.platform,
                    "backend": caps.primary_backend,
                    "cpu_cores": caps.cpu_cores,
                    "ram_gb": round(caps.ram_gb, 1),
                    "gpus": len(caps.gpus),
                    "gpu_memory_gb": sum(gpu.get('memory_mb', 0) for gpu in caps.gpus) / 1024,
                    "compute_score": round(caps.compute_score, 1),
                    "memory_score": round(caps.memory_score, 1),
                    "is_edge": caps.is_edge,
                    "power_profile": caps.power_profile,
                    "status": "online"  # TODO: monitoring réel
                }
            
            _metrics_cache.update({
                "last_update": time.time(),
                "cluster_summary": summary,
                "nodes": node_details,
                "cluster_health": "healthy" if summary["nodes"] > 0 else "no_nodes"
            })
            
        except Exception as e:
            _metrics_cache["cluster_health"] = f"error: {e}"
            
        time.sleep(5)  # Update toutes les 5s

# Démarrer le thread de monitoring
_monitor_thread = threading.Thread(target=update_metrics_cache, daemon=True)
_monitor_thread.start()

MOBILE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>VRAMancer Mobile - Cluster Hétérogène</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: #fff;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .header {
            background: rgba(0,0,0,0.3);
            padding: 1rem;
            text-align: center;
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .header h1 { font-size: 1.3rem; font-weight: 600; }
        .header .subtitle { font-size: 0.9rem; opacity: 0.8; margin-top: 0.3rem; }
        
        .container { padding: 1rem; }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .metric-card {
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .metric-value { font-size: 1.5rem; font-weight: bold; color: #4fc3f7; }
        .metric-label { font-size: 0.8rem; opacity: 0.8; margin-top: 0.3rem; }
        
        .nodes-list { margin-top: 1rem; }
        
        .node-card {
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid #4fc3f7;
        }
        
        .node-card.cuda { border-left-color: #76ff03; }
        .node-card.rocm { border-left-color: #ff5722; }
        .node-card.mps { border-left-color: #ffeb3b; }
        .node-card.edge { border-left-color: #e91e63; }
        
        .node-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .node-name { font-weight: 600; font-size: 1rem; }
        .node-badge {
            background: rgba(255,255,255,0.2);
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.7rem;
            text-transform: uppercase;
        }
        
        .node-specs {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            font-size: 0.8rem;
            opacity: 0.9;
        }
        
        .refresh-btn {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: #4fc3f7;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            color: white;
            font-size: 1.2rem;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(79, 195, 247, 0.4);
            transition: transform 0.2s;
        }
        
        .refresh-btn:active { transform: scale(0.95); }
        
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        
        .status-online { background: #4caf50; }
        .status-offline { background: #f44336; }
        .status-edge { background: #ff9800; }
        
        @media (max-width: 480px) {
            .metrics-grid { grid-template-columns: repeat(2, 1fr); }
            .node-specs { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 VRAMancer Mobile</h1>
        <div class="subtitle">Cluster Hétérogène • CUDA/ROCm/MPS/Edge</div>
    </div>
    
    <div class="container">
        <div class="metrics-grid" id="metrics-grid">
            <!-- Métriques dynamiques -->
        </div>
        
        <div class="nodes-list" id="nodes-list">
            <!-- Nœuds dynamiques -->
        </div>
    </div>
    
    <button class="refresh-btn" onclick="refreshData()">🔄</button>
    
    <script>
        async function refreshData() {
            try {
                const response = await fetch('/api/mobile/metrics');
                const data = await response.json();
                
                updateMetrics(data.cluster_summary);
                updateNodes(data.nodes);
                
                console.log('Données mises à jour:', data);
            } catch (error) {
                console.error('Erreur de mise à jour:', error);
            }
        }
        
        function updateMetrics(summary) {
            const metricsGrid = document.getElementById('metrics-grid');
            metricsGrid.innerHTML = `
                <div class="metric-card">
                    <div class="metric-value">${summary.nodes}</div>
                    <div class="metric-label">Nœuds</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${summary.total_gpus}</div>
                    <div class="metric-label">GPUs Total</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${Math.round(summary.total_ram_gb)}</div>
                    <div class="metric-label">RAM (GB)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${summary.edge_nodes || 0}</div>
                    <div class="metric-label">Edge/IoT</div>
                </div>
            `;
        }
        
        function updateNodes(nodes) {
            const nodesList = document.getElementById('nodes-list');
            let html = '';
            
            for (const [hostname, node] of Object.entries(nodes)) {
                const cardClass = node.is_edge ? 'edge' : node.backend;
                const statusClass = node.status === 'online' ? 'status-online' : 'status-offline';
                
                html += `
                    <div class="node-card ${cardClass}">
                        <div class="node-header">
                            <div class="node-name">
                                <span class="status-indicator ${statusClass}"></span>
                                ${hostname}
                            </div>
                            <div class="node-badge">${node.backend.toUpperCase()}</div>
                        </div>
                        <div class="node-specs">
                            <div>🏗️ ${node.arch}</div>
                            <div>⚡ ${node.cpu_cores} cores</div>
                            <div>🧠 ${node.ram_gb} GB RAM</div>
                            <div>🖥️ ${node.gpus} GPU(s)</div>
                            <div>📊 Score: ${node.compute_score}</div>
                            <div>🔋 ${node.power_profile}</div>
                        </div>
                    </div>
                `;
            }
            
            nodesList.innerHTML = html;
        }
        
        // Mise à jour automatique toutes les 10 secondes
        setInterval(refreshData, 10000);
        
        // Mise à jour initiale
        refreshData();
    </script>
</body>
</html>
'''

@app.route('/')
def mobile_dashboard():
    """Dashboard mobile principal."""
    return render_template_string(MOBILE_TEMPLATE)

@app.route('/api/mobile/metrics')
def mobile_metrics():
    """API REST pour métriques mobiles."""
    return jsonify(_metrics_cache)

@app.route('/api/mobile/node/<hostname>')
def mobile_node_detail(hostname):
    """Détails d'un nœud spécifique."""
    node = _metrics_cache.get("nodes", {}).get(hostname)
    if not node:
        return jsonify({"error": "Node not found"}), 404
    
    return jsonify(node)

@app.route('/api/mobile/cluster/rebalance', methods=['POST'])
def mobile_cluster_rebalance():
    """Déclenche un rééquilibrage du cluster."""
    try:
        hetero_mgr = get_heterogeneous_manager()
        # TODO: Implémenter le rééquilibrage
        return jsonify({"status": "rebalance_started", "timestamp": time.time()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Démarrage VRAMancer Mobile Dashboard")
    print("📱 Interface optimisée pour clusters hétérogènes")
    print("🌐 http://localhost:5004 (mobile/tablette)")
    app.run(host="0.0.0.0", port=5004, debug=True)