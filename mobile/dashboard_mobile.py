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
            const healthData = await health.json();
            document.getElementById('api-status').innerHTML = '✅ API Connectée';
            document.getElementById('api-details').innerHTML = `
                <div class="detail-grid">
                    <div class="detail-item"><span class="detail-label">Status:</span> ${healthData.status || 'OK'}</div>
                    <div class="detail-item"><span class="detail-label">Uptime:</span> ${healthData.uptime || 'N/A'}</div>
                </div>
            `;
            
            // Get GPU info
            try {
                const gpuResp = await fetch(`${API_BASE}/api/gpu`);
                if (gpuResp.ok) {
                    const gpuData = await gpuResp.json();
                    let gpuHtml = '';
                    if (gpuData.devices && gpuData.devices.length > 0) {
                        gpuData.devices.forEach(gpu => {
                            const memUsed = gpu.memory_used ? (gpu.memory_used / (1024*1024*1024)).toFixed(1) : 'N/A';
                            const memTotal = gpu.memory_total ? (gpu.memory_total / (1024*1024*1024)).toFixed(1) : 'N/A';
                            const memPercent = gpu.memory_used && gpu.memory_total ? 
                                ((gpu.memory_used / gpu.memory_total) * 100).toFixed(1) : 0;
                            
                            gpuHtml += `
                                <div style="margin: 8px 0;">
                                    <div><strong>🎮 ${gpu.name}</strong> (${gpu.backend})</div>
                                    <div style="margin: 4px 0;">
                                        VRAM: ${memUsed}/${memTotal} GB (${memPercent}%)
                                        <div class="progress-bar">
                                            <div class="progress-fill" style="width: ${memPercent}%"></div>
                                        </div>
                                    </div>
                                </div>
                            `;
                        });
                    } else {
                        gpuHtml = '❌ Aucun GPU détecté';
                    }
                    document.getElementById('gpu-info').innerHTML = gpuHtml;
                } else {
                    document.getElementById('gpu-info').innerHTML = '❌ Erreur récupération GPU';
                }
            } catch(e) {
                document.getElementById('gpu-info').innerHTML = '❌ GPU non disponible';
            }
            
            // Get system info
            try {
                const sysResp = await fetch(`${API_BASE}/api/system`);
                if (sysResp.ok) {
                    const sysData = await sysResp.json();
                    document.getElementById('system-resources').innerHTML = `
                        <div class="detail-grid">
                            <div class="detail-item">
                                <span class="detail-label">CPU:</span><br>
                                ${sysData.cpu_percent || 0}% (${sysData.cpu_count || 'N/A'} cores)
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">RAM:</span><br>
                                ${sysData.memory_gb || 'N/A'} GB
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">OS:</span><br>
                                ${sysData.platform || 'Unknown'}
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Backend:</span><br>
                                ${sysData.backend || 'Unknown'}
                            </div>
                        </div>
                    `;
                } else {
                    document.getElementById('system-resources').innerHTML = '❌ Erreur système';
                }
            } catch(e) {
                document.getElementById('system-resources').innerHTML = '❌ Données système non disponibles';
            }
            
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
            document.getElementById('api-details').innerHTML = 'Statut HTTP: ' + health.status;
        }
    } catch(e) { 
        document.getElementById('api-status').innerHTML = '❌ API Inaccessible';
        document.getElementById('api-details').innerHTML = 'Erreur: ' + e.message;
        document.getElementById('telemetry').textContent = 'API non disponible';
        document.getElementById('gpu-info').innerHTML = '❌ GPU non accessible';
        document.getElementById('system-resources').innerHTML = '❌ Système non accessible';
    }
    setTimeout(refresh, 8000); // Refresh every 8 seconds
}

// Test connectivité manuel complet
async function testAPI() {
    document.getElementById('test-result').innerHTML = '🔄 Test complet en cours...';
    let results = [];
    
    try {
        // Test health
        const health = await fetch(`${API_BASE}/health`);
        results.push(`Health: ${health.ok ? '✅ OK' : '❌ ' + health.status}`);
        
        // Test system
        try {
            const sys = await fetch(`${API_BASE}/api/system`);
            results.push(`System: ${sys.ok ? '✅ OK' : '❌ ' + sys.status}`);
        } catch(e) {
            results.push(`System: ❌ ${e.message}`);
        }
        
        // Test GPU
        try {
            const gpu = await fetch(`${API_BASE}/api/gpu`);
            results.push(`GPU: ${gpu.ok ? '✅ OK' : '❌ ' + gpu.status}`);
        } catch(e) {
            results.push(`GPU: ❌ ${e.message}`);
        }
        
        document.getElementById('test-result').innerHTML = results.join('<br>');
        
    } catch(e) {
        document.getElementById('test-result').innerHTML = `❌ Test échoué: ${e.message}`;
    }
}

function refreshAll() {
    document.getElementById('test-result').innerHTML = '🔄 Actualisation...';
    refresh();
    setTimeout(() => {
        document.getElementById('test-result').innerHTML = '✅ Actualisé';
    }, 1000);
}

window.onload = refresh;
"""

app = Flask(__name__)

TEMPLATE = '''
<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VRAMancer Mobile - Détaillé</title>
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
code { font-size:0.7em; line-height:1.3em; display:block; white-space:pre-wrap; word-break:break-word; }
.detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin-top: 10px; }
.detail-item { background: rgba(255,255,255,0.1); padding: 6px; border-radius: 4px; font-size: 0.85em; }
.detail-label { font-weight: bold; color: #4CAF50; }
.progress-bar { background: #333; height: 8px; border-radius: 4px; margin: 4px 0; overflow: hidden; }
.progress-fill { background: #4CAF50; height: 100%; transition: width 0.3s; }
</style>
<script>''' + MOBILE_JS + '''</script>
</head><body>
<div class="header">📱 VRAMancer Mobile - RTX 4060</div>
<div class="section">
  <div class="card">
    <b>📊 Status API VRAMancer</b>
    <div id="api-status" class="status">🔄 Vérification...</div>
    <div id="api-details" style="margin-top: 10px; font-size: 0.85em;"></div>
  </div>
  
  <div class="card">
    <b>🎮 Système GPU</b>
    <div id="gpu-info">Chargement...</div>
  </div>
  
  <div class="card">
    <b>💻 Ressources Système</b>
    <div id="system-resources">Chargement...</div>
  </div>
  
  <div class="card">
    <b>🧪 Tests & Contrôles</b>
    <button class="btn" onclick="testAPI()">Test API Complet</button>
    <button class="btn" onclick="refreshAll()">🔄 Actualiser</button>
    <div id="test-result" style="margin-top: 8px;">Cliquez pour tester</div>
  </div>
  
  <div class="card">
    <b>📡 Télémétrie Détaillée</b>
    <code id="telemetry">(chargement)</code>
  </div>
  
  <div class="card">
    <b>� Liens Rapides</b>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px;">
      <button class="btn" onclick="window.open('http://localhost:5000', '_blank')">🌐 Dashboard Web</button>
      <button class="btn" onclick="window.open('http://localhost:5030/api/status', '_blank')">📊 API Status</button>
    </div>
  </div>
  
  <div class="card">
    <b>💡 Informations</b>
    <div class="detail-grid">
      <div class="detail-item">
        <span class="detail-label">API:</span><br>
        http://localhost:5030
      </div>
      <div class="detail-item">
        <span class="detail-label">Mobile:</span><br>
        http://localhost:5003
      </div>
      <div class="detail-item">
        <span class="detail-label">GPU:</span><br>
        RTX 4060 Laptop
      </div>
      <div class="detail-item">
        <span class="detail-label">Backend:</span><br>
        PyTorch CUDA
      </div>
    </div>
  </div>
</div>
</body></html>
'''

@app.route("/")
def mobile_dashboard():
    return render_template_string(TEMPLATE)

@app.route('/telemetry')
def mobile_telemetry_proxy():
    """Télémétrie détaillée pour interface mobile"""
    try:
        telemetry_data = []
        
        # 1. Status API
        try:
            r = requests.get('http://localhost:5030/api/status', timeout=2)
            if r.status_code == 200:
                data = r.json()
                telemetry_data.append("=== API STATUS ===")
                telemetry_data.append(f"Status: OK")
                telemetry_data.append(f"Data: {json.dumps(data, indent=2)}")
            else:
                telemetry_data.append(f"API Status: Error {r.status_code}")
        except Exception as e:
            telemetry_data.append(f"API Status: Non connectée - {str(e)}")
        
        # 2. Système
        try:
            sys_r = requests.get('http://localhost:5030/api/system', timeout=2)
            if sys_r.status_code == 200:
                sys_data = sys_r.json()
                telemetry_data.append("\n=== SYSTÈME ===")
                telemetry_data.append(f"Platform: {sys_data.get('platform', 'Unknown')}")
                telemetry_data.append(f"CPU: {sys_data.get('cpu_count', 'N/A')} cores @ {sys_data.get('cpu_percent', 0)}%")
                telemetry_data.append(f"Memory: {sys_data.get('memory_gb', 'N/A')} GB")
                telemetry_data.append(f"Backend: {sys_data.get('backend', 'Unknown')}")
        except Exception as e:
            telemetry_data.append(f"\nSystème: Erreur - {str(e)}")
        
        # 3. GPU
        try:
            gpu_r = requests.get('http://localhost:5030/api/gpu', timeout=2)
            if gpu_r.status_code == 200:
                gpu_data = gpu_r.json()
                telemetry_data.append("\n=== GPU ===")
                if gpu_data.get('devices'):
                    for i, gpu in enumerate(gpu_data['devices']):
                        telemetry_data.append(f"GPU {i}: {gpu.get('name', 'Unknown')}")
                        telemetry_data.append(f"  Backend: {gpu.get('backend', 'Unknown')}")
                        if gpu.get('memory_total'):
                            mem_used = gpu.get('memory_used', 0) / (1024**3)
                            mem_total = gpu.get('memory_total', 0) / (1024**3)
                            telemetry_data.append(f"  VRAM: {mem_used:.1f}/{mem_total:.1f} GB")
                else:
                    telemetry_data.append("Aucun GPU détecté")
        except Exception as e:
            telemetry_data.append(f"\nGPU: Erreur - {str(e)}")
        
        # 4. Memory (si disponible)
        try:
            mem_r = requests.get('http://localhost:5030/api/memory', timeout=2)
            if mem_r.status_code == 200:
                mem_data = mem_r.json()
                telemetry_data.append(f"\n=== MÉMOIRE VRAMANCER ===")
                telemetry_data.append(f"Blocs actifs: {len(mem_data.get('blocks', {}))}")
                telemetry_data.append(f"Status: {mem_data.get('status', 'Unknown')}")
        except:
            pass
            
        return "\n".join(telemetry_data), 200, {'Content-Type':'text/plain'}
        
    except Exception as e:
        return f"ERREUR TÉLÉMÉTRIE: {str(e)}", 200, {'Content-Type':'text/plain'}

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