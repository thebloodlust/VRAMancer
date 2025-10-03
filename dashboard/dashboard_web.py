# dashboard/dashboard_web.py

from flask import Flask, render_template_string, request, jsonify
from utils.gpu_utils import get_available_gpus
import subprocess

app = Flask(__name__)

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
    <meta http-equiv="refresh" content="5">
    <style>
        body { background-color: #121212; color: #eee; font-family: Arial; padding: 20px; }
        .gpu { background: #1e1e1e; padding: 15px; margin-bottom: 10px; border-radius: 8px; }
        .bar { height: 20px; background: #00BFFF; margin-top: 5px; border-radius: 4px; }
        select, button { margin-top: 10px; padding: 5px; font-size: 1em; }
    </style>
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

    {% for gpu in gpus %}
    <div class="gpu">
        <h2>🎮 {{ gpu.name }}</h2>
        <p>💾 VRAM: {{ gpu.used_vram_mb }} / {{ gpu.total_vram_mb }} MB</p>
        <div class="bar" style="width: {{ gpu.used_vram_mb / gpu.total_vram_mb * 100 }}%;"></div>
        <p>📡 Statut: {{ "✅ Disponible" if gpu.is_available else "❌ Indisponible" }}</p>
    </div>
    {% endfor %}
</body>
</html>
"""

@app.route("/")
def dashboard():
    gpus = get_available_gpus()
    return render_template_string(TEMPLATE, gpus=gpus)

@app.route("/api/memory")
def api_memory():
    if HIER_MEMORY is None:
        return jsonify({"error": "no memory manager"}), 404
    return jsonify(HIER_MEMORY.summary())

@app.route("/switch")
def switch():
    mode = request.args.get("mode", "cli")
    subprocess.Popen(["python3", "launcher.py", "--mode", mode])
    return f"<p>Lancement de l’interface {mode}…</p><meta http-equiv='refresh' content='2;url=/' />"

def launch():
    app.run(debug=False, host="0.0.0.0", port=5000)
