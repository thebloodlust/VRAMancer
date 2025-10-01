"""
API no-code pour générer, exécuter et monitorer des workflows IA.
- REST endpoints pour créer, lister, exécuter, exporter des workflows
- Templates de tâches (inférence, offload, migration, monitoring, auto-scale)
- Exécution asynchrone, logs, état, rollback
"""
from flask import Flask, request, jsonify
import threading
import time
import uuid

app = Flask(__name__)
WORKFLOWS = {}
RESULTS = {}

TEMPLATES = {
    "inference": {"params": ["model", "input"]},
    "offload": {"params": ["block_id", "target"]},
    "monitor": {"params": ["resource", "threshold"]},
    "auto_scale": {"params": ["min_nodes", "max_nodes"]},
    "migration": {"params": ["block_id", "src", "dst"]},
}

@app.route("/api/workflows/templates")
def get_templates():
    return jsonify(TEMPLATES)

@app.route("/api/workflows", methods=["POST"])
def create_workflow():
    data = request.json
    wid = str(uuid.uuid4())
    WORKFLOWS[wid] = {"id": wid, "tasks": data.get("tasks", []), "status": "created", "logs": []}
    return jsonify(WORKFLOWS[wid])

@app.route("/api/workflows/<wid>")
def get_workflow(wid):
    return jsonify(WORKFLOWS.get(wid, {}))

@app.route("/api/workflows/<wid>/run", methods=["POST"])
def run_workflow(wid):
    def runner():
        wf = WORKFLOWS[wid]
        wf["status"] = "running"
        for i, task in enumerate(wf["tasks"]):
            time.sleep(1)  # Simule exécution
            wf["logs"].append(f"Tâche {i+1}: {task['type']} exécutée")
        wf["status"] = "finished"
        RESULTS[wid] = {"result": "ok", "logs": wf["logs"]}
    threading.Thread(target=runner).start()
    return jsonify({"ok": True, "status": "started"})

@app.route("/api/workflows/<wid>/logs")
def get_logs(wid):
    wf = WORKFLOWS.get(wid, {})
    return jsonify(wf.get("logs", []))

@app.route("/api/workflows/<wid>/export")
def export_workflow(wid):
    wf = WORKFLOWS.get(wid, {})
    return jsonify({"id": wid, "tasks": wf.get("tasks", [])})

if __name__ == "__main__":
    app.run(port=5020, debug=True)
