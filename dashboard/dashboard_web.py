# dashboard/dashboard_web.py

from flask import Flask, render_template_string
from utils.gpu_utils import get_available_gpus

app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VRAMancer Web</title>
    <meta http-equiv="refresh" content="2">
    <style>
        body { background-color: #121212; color: #eee; font-family: Arial; padding: 20px; }
        .gpu { background: #1e1e1e; padding: 15px; margin-bottom: 10px; border-radius: 8px; }
        .bar { height: 20px; background: #00BFFF; margin-top: 5px; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>🧠 VRAMancer Web Dashboard</h1>
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

def launch():
    app.run(debug=False, host="0.0.0.0", port=5000)
