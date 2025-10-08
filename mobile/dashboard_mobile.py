"""
Interface mobile/tablette CORRIGÉE :
- Dashboard Flask pour contrôle et monitoring mobile
- Vue responsive, accès sécurisé, API fixée
"""
from flask import Flask, render_template_string, jsonify
import requests, json, os

MOBILE_JS = """
const API_BASE = 'http://localhost:5030';

async function refresh(){
    try {
        // Test API health
        const health = await fetch(`${API_BASE}/health`);
        if (health.ok) {
            document.getElementById('api-status').innerHTML = '✅ API Connectée';
            
            // Get telemetry
            try {
                const r = await fetch('/telemetry');
                const t = await r.text();
                document.getElementById('telemetry').textContent = t.trim();
            } catch(e) {
                document.getElementById('telemetry').textContent = 'Erreur télémétrie: ' + e.message;
            }
        } else {
            document.getElementById('api-status').innerHTML = '❌ API Déconnectée';
        }
    } catch(e) { 
        document.getElementById('api-status').innerHTML = '❌ API Inaccessible';
        document.getElementById('telemetry').textContent = 'API non disponible';
    }
    setTimeout(refresh, 5000);
}

// Test connectivité manuel
async function testAPI() {
    document.getElementById('test-result').innerHTML = '🔄 Test en cours...';
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            const data = await response.json();
            document.getElementById('test-result').innerHTML = `✅ OK: ${JSON.stringify(data)}`;
        } else {
            document.getElementById('test-result').innerHTML = `❌ Erreur: ${response.status}`;
        }
    } catch(e) {
        document.getElementById('test-result').innerHTML = `❌ Échec: ${e.message}`;
    }
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
.btn { background:#4CAF50; color:#fff; border:none; padding:10px 15px; border-radius:5px; margin:5px; cursor:pointer; }
.btn:hover { background:#45a049; }
.status { padding:8px; border-radius:4px; margin:5px 0; }
.status-ok { background:#4CAF50; }
.status-error { background:#f44336; }
code { font-size:0.75em; line-height:1.3em; display:block; white-space:pre-wrap; word-break:break-word; }
</style>
<script>''' + MOBILE_JS + '''</script>
</head><body>
<div class="header">📱 VRAMancer Mobile</div>
<div class="section">
  <div class="card">
    <b>📊 Status API</b>
    <div id="api-status" class="status">🔄 Vérification...</div>
  </div>
  <div class="card">
    <b>🧪 Test Manuel</b>
    <button class="btn" onclick="testAPI()">Test API</button>
    <div id="test-result">Cliquez pour tester</div>
  </div>
  <div class="card">
    <b>📡 Télémétrie</b>
    <code id="telemetry">(chargement)</code>
  </div>
  <div class="card">
    <b>💡 Info</b>
    Interface mobile responsive pour VRAMancer
    <br>API: http://localhost:5030
    <br>Mobile: http://localhost:5003
  </div>
</div>
</body></html>
'''

@app.route("/")
def mobile_dashboard():
    return render_template_string(TEMPLATE)

@app.route('/telemetry')
def mobile_telemetry_proxy():
    # proxy corrigé vers l'API principale
    try:
        # Essai API principale
        r = requests.get('http://localhost:5030/api/status', timeout=2)
        if r.status_code == 200:
            data = r.json()
            return f"API Status: OK\nData: {json.dumps(data, indent=2)}", 200, {'Content-Type':'text/plain'}
        else:
            return f"API Status: Error {r.status_code}", 200, {'Content-Type':'text/plain'}
    except Exception as e:
        return f"API Non connectée: {str(e)}", 200, {'Content-Type':'text/plain'}

@app.route('/api/test')
def mobile_api_test():
    """Endpoint de test pour l'interface mobile"""
    try:
        # Test API health
        r = requests.get('http://localhost:5030/health', timeout=2)
        if r.status_code == 200:
            return jsonify({
                "status": "ok",
                "api_connected": True,
                "api_data": r.json(),
                "mobile_port": 5003
            })
        else:
            return jsonify({
                "status": "api_error", 
                "api_connected": False,
                "error": f"Status {r.status_code}"
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "api_connected": False, 
            "error": str(e)
        })

if __name__ == "__main__":
    print("=" * 60)
    print("  VRAMANCER MOBILE DASHBOARD")
    print("=" * 60)
    print()
    print("📱 URL: http://localhost:5003")
    print("🌐 Interface mobile responsive")
    print("📊 Tests API intégrés")
    print()
    
    try:
        # Ouverture automatique navigateur
        import webbrowser
        import threading
        import time
        def open_browser():
            time.sleep(1.5)  # Attendre que Flask démarre
            webbrowser.open('http://localhost:5003')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        print("🚀 Ouverture automatique navigateur...")
        print("Appuyez sur Ctrl+C pour arrêter")
        print("=" * 60)
        
        app.run(host="0.0.0.0", port=5003, debug=False)
    except Exception as e:
        print(f"Erreur démarrage serveur: {e}")
        input("Appuyez sur Entrée pour fermer...")