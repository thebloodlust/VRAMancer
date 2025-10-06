"""
Interface mobile/tablette :
- Dashboard Flask pour contrôle et monitoring mobile
- Vue responsive, accès sécurisé
"""
from flask import Flask, render_template_string
import requests, json

MOBILE_JS = """
async function refresh(){
    try {
        const r = await fetch('/telemetry');
        const t = await r.text();
        document.getElementById('telemetry').textContent = t.trim();
    } catch(e) { /* ignore */ }
    setTimeout(refresh, 4000);
}
window.onload = refresh;
"""

app = Flask(__name__)

TEMPLATE = '''
<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VRAMancer Mobile</title>
<style>
body { font-family: system-ui, sans-serif; margin:0; padding:0; background:#111; color:#eee; }
.header { background:#1976D2; color:#fff; padding:1em; text-align:center; font-weight:600; }
.section { margin:1em; }
.card { background:#1e1e1e; margin:0.8em 0; padding:0.9em; border-radius:10px; box-shadow:0 2px 4px #0006; }
code { font-size:0.75em; line-height:1.3em; display:block; white-space:pre-wrap; word-break:break-word; }
</style>
<script>''' + MOBILE_JS + '''</script>
</head><body>
<div class="header">VRAMancer Mobile</div>
<div class="section">
  <div class="card"><b>Supervision compacte</b><code id="telemetry">(chargement)</code></div>
  <div class="card">Astuce: utilisez /api/telemetry.bin pour un flux binaire.</div>
</div>
</body></html>
'''

@app.route("/")
def mobile_dashboard():
    return render_template_string(TEMPLATE)

@app.route('/telemetry')
def mobile_telemetry_proxy():
    # proxy minimal vers la supervision texte
    try:
        r = requests.get('http://localhost:5010/api/telemetry.txt', timeout=1)
        return r.text, 200, {'Content-Type':'text/plain'}
    except Exception as e:
        return f"erreur: {e}", 500, {'Content-Type':'text/plain'}

if __name__ == "__main__":
    app.run(port=5003, debug=True)
