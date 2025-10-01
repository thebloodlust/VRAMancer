from flask import Flask, render_template_string, request, jsonify
import threading
import time

app = Flask(__name__)

# Simule l’état du cluster
cluster_state = {
    "nodes": [
        {"host": "nodeA", "os": "Linux", "vram": 8192, "cpu": 8, "status": "active"},
        {"host": "nodeB", "os": "Windows", "vram": 16384, "cpu": 16, "status": "active"},
        {"host": "nodeC", "os": "macOS", "vram": 4096, "cpu": 4, "status": "idle"}
    ],
    "logs": []
}

# Template HTML minimaliste
TEMPLATE = """
<!DOCTYPE html>
<html lang='fr'>
<head>
    <meta charset='UTF-8'>
    <title>VRAMancer Web Dashboard</title>
    <style>
        body { font-family: Arial; background: #181818; color: #eee; }
        .node { border: 1px solid #444; margin: 8px; padding: 8px; border-radius: 8px; background: #222; }
        .active { color: #00FF00; }
        .idle { color: #FFD700; }
        .log { font-size: 0.9em; color: #aaa; }
        button { background: #00BFFF; color: #fff; border: none; padding: 8px; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>VRAMancer Cluster Dashboard</h1>
    <h2>Nodes</h2>
    {% for node in nodes %}
        <div class='node {{node.status}}'>
            <b>{{node.host}}</b> ({{node.os}}) | VRAM: {{node.vram}} MB | CPU: {{node.cpu}} | Status: <span class='{{node.status}}'>{{node.status}}</span>
            <form method='post' action='/control'>
                <input type='hidden' name='host' value='{{node.host}}'>
                <button name='action' value='activate'>Activate</button>
                <button name='action' value='deactivate'>Deactivate</button>
            </form>
        </div>
    {% endfor %}
    <h2>Logs</h2>
    <div>
        {% for log in logs %}
            <div class='log'>{{log}}</div>
        {% endfor %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(TEMPLATE, nodes=cluster_state["nodes"], logs=cluster_state["logs"])

@app.route("/control", methods=["POST"])
def control():
    host = request.form.get("host")
    action = request.form.get("action")
    for node in cluster_state["nodes"]:
        if node["host"] == host:
            node["status"] = "active" if action == "activate" else "idle"
            cluster_state["logs"].append(f"[{time.strftime('%H:%M:%S')}] {host} set to {node['status']}")
    return index()

@app.route("/api/state", methods=["GET"])
def api_state():
    return jsonify(cluster_state)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
