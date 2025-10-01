"""
API et protocole custom VRAMancer pour supervision avancée des nœuds.
- État en ligne/hors ligne, type, icône, CPU, RAM, GPU, OS, connexion (USB4, Ethernet, WiFi)
- Historique, alertes, actions distantes
- REST + WebSocket
"""
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
import time
import random

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

NODES = [
    {"id": "raspberrypi", "type": "edge", "icon": "raspberry.svg", "status": "online", "cpu": "ARM", "ram": 1024, "gpu": "None", "os": "Linux", "conn": "WiFi", "usb4": False, "last_seen": time.time()},
    {"id": "jetson", "type": "edge", "icon": "jetson.svg", "status": "offline", "cpu": "ARM", "ram": 4096, "gpu": "Nvidia", "os": "Linux", "conn": "Ethernet", "usb4": True, "last_seen": time.time()-3600},
    {"id": "workstation", "type": "standard", "icon": "standard.svg", "status": "online", "cpu": "x86_64", "ram": 32768, "gpu": "RTX 4090", "os": "Windows", "conn": "USB4", "usb4": True, "last_seen": time.time()},
]

HISTORY = []

@app.route("/api/nodes")
def get_nodes():
    return jsonify(NODES)

@app.route("/api/nodes/<node_id>")
def get_node(node_id):
    node = next((n for n in NODES if n["id"] == node_id), None)
    return jsonify(node or {})

@app.route("/api/nodes/<node_id>/action", methods=["POST"])
def node_action(node_id):
    action = request.json.get("action")
    HISTORY.append({"node": node_id, "action": action, "timestamp": time.time()})
    # Simuler une action distante
    return jsonify({"ok": True, "action": action})

@app.route("/api/history")
def get_history():
    return jsonify(HISTORY)

@socketio.on("subscribe")
def handle_subscribe(data):
    emit("nodes", NODES)

@socketio.on("ping")
def handle_ping(data):
    node_id = data.get("node_id")
    # Simuler un ping
    emit("pong", {"node_id": node_id, "status": random.choice(["online", "offline"])})

if __name__ == "__main__":
    socketio.run(app, port=5010, debug=True)
