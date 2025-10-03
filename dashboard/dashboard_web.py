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

    <h2>Hiérarchie Mémoire</h2>
    {% if memory %}
    <table border="1" cellpadding="6" cellspacing="0">
        <tr><th>Tier</th><th>Count</th></tr>
        {% for tier, count in memory.tiers.items() %}
        <tr><td>{{ tier }}</td><td>{{ count }}</td></tr>
        {% endfor %}
    </table>
    <h3>Blocs</h3>
    <table border="1" cellpadding="4" cellspacing="0">
        <tr><th>ID</th><th>Tier</th><th>Size(MB)</th><th>Access</th><th>Dernier accès</th><th>Actions</th></tr>
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
    </table>
    {% else %}
    <p>(Pas de données mémoire)</p>
    {% endif %}

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

def launch():
    app.run(debug=False, host="0.0.0.0", port=5000)
