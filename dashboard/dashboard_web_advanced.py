from flask import Flask, render_template_string, request, jsonify
import threading
import time
import requests
import json

app = Flask(__name__)

# État du cluster corrigé avec API connection
cluster_state = {
    "nodes": [],
    "logs": [],
    "api_connected": False,
    "last_update": None
}

def update_from_api():
    """Met à jour l'état depuis l'API VRAMancer"""
    try:
        # Test API connection
        health_response = requests.get('http://localhost:5030/health', timeout=2)
        if health_response.status_code == 200:
            cluster_state["api_connected"] = True
            cluster_state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Get nodes info
            try:
                nodes_response = requests.get('http://localhost:5030/api/nodes', timeout=2)
                if nodes_response.status_code == 200:
                    api_nodes = nodes_response.json()
                    # Format pour affichage
                    if isinstance(api_nodes, dict) and "nodes" in api_nodes:
                        cluster_state["nodes"] = api_nodes["nodes"]
                    else:
                        cluster_state["nodes"] = api_nodes if isinstance(api_nodes, list) else []
                else:
                    cluster_state["nodes"] = [{"host": "API-Node", "status": "error", "info": f"Status {nodes_response.status_code}"}]
            except Exception as e:
                cluster_state["nodes"] = [{"host": "API-Error", "status": "error", "info": str(e)}]
                
        else:
            cluster_state["api_connected"] = False
            cluster_state["nodes"] = [{"host": "API-Disconnected", "status": "disconnected", "info": f"HTTP {health_response.status_code}"}]
            
    except Exception as e:
        cluster_state["api_connected"] = False
        cluster_state["nodes"] = [{"host": "API-Unavailable", "status": "unavailable", "info": str(e)}]
        cluster_state["logs"].append(f"[{time.strftime('%H:%M:%S')}] API Error: {str(e)}")

# Mise à jour initiale
update_from_api()

# Template HTML avancé corrigé
TEMPLATE = """
<!DOCTYPE html>
<html lang='fr'>
<head>
    <meta charset='UTF-8'>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VRAMancer Dashboard Web Avancé</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: #eee; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; }
        .status-bar { display: flex; justify-content: space-between; background: rgba(0,0,0,0.3); padding: 15px; border-radius: 10px; margin-bottom: 20px; }
        .node { border: 1px solid #444; margin: 10px 0; padding: 15px; border-radius: 10px; background: rgba(255,255,255,0.1); }
        .active { border-left: 4px solid #00FF00; }
        .idle { border-left: 4px solid #FFD700; }
        .error { border-left: 4px solid #FF4444; }
        .disconnected { border-left: 4px solid #FF8800; }
        .unavailable { border-left: 4px solid #AA4444; }
        .log { font-size: 0.9em; color: #aaa; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; margin: 5px 0; }
        button { background: #4CAF50; color: #fff; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer; margin: 2px; }
        button:hover { background: #45a049; transform: translateY(-1px); }
        .refresh-btn { background: #2196F3; }
        .control-panel { background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px; margin: 20px 0; }
        .api-status { padding: 5px 10px; border-radius: 5px; }
        .api-ok { background: #4CAF50; }
        .api-error { background: #f44336; }
    </style>
    <script>
        function refreshPage() {
            window.location.reload();
        }
        
        function autoRefresh() {
            setTimeout(() => {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('node-count').textContent = data.node_count || 0;
                        document.getElementById('api-status').textContent = data.api_connected ? '✅ Connectée' : '❌ Déconnectée';
                        document.getElementById('last-update').textContent = data.last_update || 'Jamais';
                    })
                    .catch(error => {
                        document.getElementById('api-status').textContent = '❌ Error';
                    });
                autoRefresh();
            }, 10000);
        }
        
        window.onload = autoRefresh;
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌐 VRAMancer Dashboard Web Avancé</h1>
            <p>Supervision cluster hétérogène en temps réel</p>
        </div>
        
        <div class="status-bar">
            <div>API Status: <span id="api-status" class="api-status {{ 'api-ok' if api_connected else 'api-error' }}">{{ '✅ Connectée' if api_connected else '❌ Déconnectée' }}</span></div>
            <div>Nodes: <span id="node-count">{{ nodes|length }}</span></div>
            <div>Dernière MàJ: <span id="last-update">{{ last_update or 'Jamais' }}</span></div>
        </div>
        
        <div class="control-panel">
            <button onclick="refreshPage()" class="refresh-btn">🔄 Actualiser</button>
            <button onclick="fetch('/api/refresh').then(() => location.reload())">📡 Sync API</button>
            <button onclick="window.open('/api/test', '_blank')">🧪 Test API</button>
        </div>
        
        <h2>📊 Cluster Nodes ({{ nodes|length }})</h2>
        {% if nodes %}
        {% for node in nodes %}
            <div class='node {{ node.status or "unknown" }}'>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <b>{{ node.host or node.name or "Unknown" }}</b>
                        {% if node.os %} ({{ node.os }}) {% endif %}
                        {% if node.vram %} | VRAM: {{ node.vram }} MB {% endif %}
                        {% if node.cpu %} | CPU: {{ node.cpu }} {% endif %}
                        {% if node.info %} | Info: {{ node.info }} {% endif %}
                    </div>
                    <div>
                        Status: <span style="font-weight: bold;">{{ node.status or "unknown" }}</span>
                    </div>
                </div>
                {% if node.status == 'error' or node.status == 'disconnected' or node.status == 'unavailable' %}
                <div style="margin-top: 10px; padding: 8px; background: rgba(255,0,0,0.1); border-radius: 5px;">
                    ⚠️ Problème détecté: {{ node.info or "Erreur inconnue" }}
                </div>
                {% endif %}
            </div>
        {% endfor %}
        {% else %}
        <div class='node error'>
            <b>Aucun node détecté</b>
            <div>API Status: {{ '✅ Connectée' if api_connected else '❌ Déconnectée' }}</div>
            {% if not api_connected %}
            <div style="margin-top: 10px; padding: 8px; background: rgba(255,0,0,0.1); border-radius: 5px;">
                💡 Vérifiez que l'API VRAMancer est lancée sur le port 5030
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <h2>📝 Logs Système ({{ logs|length }})</h2>
        <div>
            {% if logs %}
            {% for log in logs[-10:] %}
                <div class='log'>{{ log }}</div>
            {% endfor %}
            {% else %}
            <div class='log'>Aucun log disponible</div>
            {% endif %}
        </div>
        
        <div class="control-panel">
            <h3>🔧 Actions</h3>
            <button onclick="fetch('/api/refresh').then(() => location.reload())">🔄 Synchroniser API</button>
            <button onclick="fetch('/api/clear-logs', {method: 'POST'}).then(() => location.reload())">🧹 Effacer Logs</button>
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    # Mise à jour avant affichage
    update_from_api()
    return render_template_string(TEMPLATE, 
                                nodes=cluster_state["nodes"], 
                                logs=cluster_state["logs"],
                                api_connected=cluster_state["api_connected"],
                                last_update=cluster_state["last_update"])

@app.route("/api/status", methods=["GET"])
def api_status():
    """Endpoint pour mise à jour AJAX"""
    update_from_api()
    return jsonify({
        "node_count": len(cluster_state["nodes"]),
        "api_connected": cluster_state["api_connected"],
        "last_update": cluster_state["last_update"],
        "logs_count": len(cluster_state["logs"])
    })

@app.route("/api/refresh", methods=["GET", "POST"])
def api_refresh():
    """Force refresh depuis API"""
    update_from_api()
    return jsonify({"status": "refreshed", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})

@app.route("/api/clear-logs", methods=["POST"])
def clear_logs():
    """Efface les logs"""
    cluster_state["logs"].clear()
    cluster_state["logs"].append(f"[{time.strftime('%H:%M:%S')}] Logs effacés")
    return jsonify({"status": "logs_cleared"})

@app.route("/api/test", methods=["GET"])
def api_test():
    """Page de test API"""
    try:
        r = requests.get('http://localhost:5030/health', timeout=2)
        if r.status_code == 200:
            return jsonify({
                "dashboard_status": "ok",
                "api_status": "connected",
                "api_data": r.json(),
                "port": 5000
            })
        else:
            return jsonify({
                "dashboard_status": "ok", 
                "api_status": "error",
                "error": f"HTTP {r.status_code}"
            })
    except Exception as e:
        return jsonify({
            "dashboard_status": "ok",
            "api_status": "disconnected",
            "error": str(e)
        })

@app.route("/api/state", methods=["GET"])
def api_state():
    return jsonify(cluster_state)

if __name__ == "__main__":
    print("=" * 60)
    print("  VRAMANCER DASHBOARD WEB AVANCÉ")
    print("=" * 60)
    print()
    print("🌐 URL: http://localhost:5000")
    print("📊 Supervision cluster en temps réel")
    print("🔄 Auto-refresh activé")
    print()
    
    try:
        # Test initial API
        update_from_api()
        
        # Ouverture automatique navigateur
        import webbrowser
        import threading
        def open_browser():
            time.sleep(1.5)  # Attendre que Flask démarre
            webbrowser.open('http://localhost:5000')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        print("🚀 Ouverture automatique navigateur...")
        print("Appuyez sur Ctrl+C pour arrêter")
        print("=" * 60)
        
        app.run(host="0.0.0.0", port=5000, debug=False)
    except Exception as e:
        print(f"Erreur démarrage serveur: {e}")
        input("Appuyez sur Entrée pour fermer...")