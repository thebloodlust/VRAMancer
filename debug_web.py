#!/usr/bin/env python3
"""
VRAMancer Web Debug - Diagnostic complet et réparation automatique
"""

import os
import sys
import time
import subprocess
import threading
import webbrowser
import traceback
from pathlib import Path

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_status(message, status="INFO"):
    colors = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️", "DEBUG": "🔍"}
    print(f"{colors.get(status, 'ℹ️')} {message}")

def install_package(package_name, import_name=None):
    """Installation automatique d'un package"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print_status(f"{package_name} déjà installé", "SUCCESS")
        return True
    except ImportError:
        print_status(f"Installation de {package_name}...", "INFO")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package_name, 
                "--quiet", "--disable-pip-version-check"
            ])
            print_status(f"{package_name} installé avec succès", "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            print_status(f"Erreur installation {package_name}: {e}", "ERROR")
            return False

def test_api_connection():
    """Test de connexion à l'API"""
    print_section("TEST DE CONNEXION API")
    
    api_base = os.environ.get('VRM_API_BASE', 'http://localhost:5030')
    print_status(f"API configurée: {api_base}", "INFO")
    
    try:
        import requests
        print_status("Module requests disponible", "SUCCESS")
        
        # Test health endpoint
        print_status("Test endpoint /health...", "DEBUG")
        response = requests.get(f"{api_base}/health", timeout=5)
        if response.status_code == 200:
            print_status(f"Health check OK: {response.json()}", "SUCCESS")
        else:
            print_status(f"Health check failed: {response.status_code}", "ERROR")
            return False
        
        # Test autres endpoints
        endpoints = ["/api/status", "/api/gpu/info", "/api/nodes", "/api/telemetry.bin"]
        for endpoint in endpoints:
            try:
                resp = requests.get(f"{api_base}{endpoint}", timeout=3)
                if resp.status_code == 200:
                    print_status(f"✅ {endpoint}: OK", "SUCCESS")
                else:
                    print_status(f"⚠️ {endpoint}: {resp.status_code}", "WARNING")
            except Exception as e:
                print_status(f"❌ {endpoint}: {e}", "ERROR")
        
        return True
        
    except ImportError:
        print_status("Module requests manquant", "ERROR")
        return False
    except Exception as e:
        print_status(f"Erreur test API: {e}", "ERROR")
        print_status("L'API n'est peut-être pas démarrée", "WARNING")
        print_status("Lancez d'abord: python start_api.py", "INFO")
        return False

def test_flask_setup():
    """Test de configuration Flask"""
    print_section("TEST CONFIGURATION FLASK")
    
    try:
        from flask import Flask, render_template_string, jsonify
        print_status("Flask importé avec succès", "SUCCESS")
        
        # Test création app
        app = Flask(__name__)
        print_status("Application Flask créée", "SUCCESS")
        
        @app.route('/test')
        def test_route():
            return jsonify({"status": "ok", "message": "Test route fonctionne"})
        
        print_status("Route de test définie", "SUCCESS")
        return app
        
    except ImportError as e:
        print_status(f"Erreur import Flask: {e}", "ERROR")
        return None
    except Exception as e:
        print_status(f"Erreur configuration Flask: {e}", "ERROR")
        return None

def create_debug_web_server():
    """Création d'un serveur web de debug"""
    print_section("CRÉATION SERVEUR WEB DEBUG")
    
    try:
        from flask import Flask, jsonify, request
        import requests
        
        app = Flask(__name__)
        
        # Variables globales pour debug
        debug_info = {
            "requests_count": 0,
            "last_error": None,
            "api_status": "unknown"
        }
        
        @app.route('/')
        def debug_dashboard():
            api_base = os.environ.get('VRM_API_BASE', 'http://localhost:5030')
            return f'''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VRAMancer - Debug Dashboard Web</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #1a1a1a;
            color: #fff;
            font-family: 'Consolas', 'Monaco', monospace;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 20px 0;
            border-bottom: 2px solid #00bfff;
            margin-bottom: 30px;
        }}
        .debug-section {{
            background: #2d2d2d;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .debug-section h3 {{
            color: #00bfff;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .status-ok {{ color: #28a745; }}
        .status-error {{ color: #dc3545; }}
        .status-warning {{ color: #ffc107; }}
        .btn {{
            background: #007acc;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            font-size: 14px;
        }}
        .btn:hover {{ background: #005a9e; }}
        .log-area {{
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 5px;
            padding: 15px;
            font-family: monospace;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #444;
        }}
        .metric:last-child {{ border-bottom: none; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 VRAMancer - Debug Dashboard Web</h1>
            <p>Diagnostic complet et monitoring en temps réel</p>
        </div>
        
        <div class="debug-section">
            <h3>🌐 Configuration Serveur</h3>
            <div class="metric">
                <span>Serveur Web:</span>
                <span class="status-ok">✅ Actif sur port 8080</span>
            </div>
            <div class="metric">
                <span>API Backend:</span>
                <span id="api-status">🔄 Vérification...</span>
            </div>
            <div class="metric">
                <span>Python Version:</span>
                <span>{sys.version.split()[0]}</span>
            </div>
            <div class="metric">
                <span>API Base URL:</span>
                <span>{api_base}</span>
            </div>
        </div>
        
        <div class="debug-section">
            <h3>🧪 Tests API</h3>
            <button class="btn" onclick="testAllEndpoints()">Test Tous les Endpoints</button>
            <button class="btn" onclick="testConnectivity()">Test Connectivité</button>
            <button class="btn" onclick="clearLog()">Effacer Log</button>
            <div id="api-results" class="log-area">Cliquez sur "Test Tous les Endpoints" pour commencer...</div>
        </div>
        
        <div class="debug-section">
            <h3>📊 Monitoring Temps Réel</h3>
            <div id="monitoring-data" class="log-area">Chargement des données de monitoring...</div>
            <button class="btn" onclick="refreshMonitoring()">🔄 Actualiser</button>
        </div>
        
        <div class="debug-section">
            <h3>🔧 Actions Debug</h3>
            <button class="btn" onclick="downloadDebugInfo()">📥 Télécharger Debug Info</button>
            <button class="btn" onclick="testNetworkLatency()">⚡ Test Latence Réseau</button>
            <button class="btn" onclick="window.open('/debug/info', '_blank')">📋 Info Système</button>
        </div>
        
        <div class="debug-section">
            <h3>📝 Log Temps Réel</h3>
            <div id="live-log" class="log-area"></div>
        </div>
    </div>

    <script>
        const API_BASE = '{api_base}';
        let logCounter = 0;
        
        function log(message, type = 'info') {{
            const timestamp = new Date().toLocaleTimeString();
            const colors = {{
                'info': '#00bfff',
                'success': '#28a745',
                'error': '#dc3545',
                'warning': '#ffc107'
            }};
            
            const logArea = document.getElementById('live-log');
            logArea.innerHTML += `<span style="color: ${{colors[type] || '#fff'}}">[${{timestamp}}] ${{message}}</span>\\n`;
            logArea.scrollTop = logArea.scrollHeight;
            
            if (++logCounter > 100) {{
                const lines = logArea.innerHTML.split('\\n');
                logArea.innerHTML = lines.slice(-50).join('\\n');
                logCounter = 50;
            }}
        }}
        
        async function testAllEndpoints() {{
            const resultsArea = document.getElementById('api-results');
            resultsArea.innerHTML = 'Test en cours...\\n';
            
            const endpoints = [
                '/health',
                '/api/status', 
                '/api/gpu/info',
                '/api/nodes',
                '/api/telemetry.bin'
            ];
            
            for (const endpoint of endpoints) {{
                try {{
                    const startTime = performance.now();
                    const response = await fetch(`${{API_BASE}}${{endpoint}}`);
                    const endTime = performance.now();
                    const latency = Math.round(endTime - startTime);
                    
                    if (response.ok) {{
                        const data = await response.json();
                        resultsArea.innerHTML += `✅ ${{endpoint}}: OK (${{latency}}ms)\\n`;
                        resultsArea.innerHTML += `   Réponse: ${{JSON.stringify(data, null, 2)}}\\n\\n`;
                        log(`Endpoint ${{endpoint}}: OK (${{latency}}ms)`, 'success');
                    }} else {{
                        resultsArea.innerHTML += `❌ ${{endpoint}}: HTTP ${{response.status}}\\n\\n`;
                        log(`Endpoint ${{endpoint}}: Erreur ${{response.status}}`, 'error');
                    }}
                }} catch (error) {{
                    resultsArea.innerHTML += `❌ ${{endpoint}}: ${{error.message}}\\n\\n`;
                    log(`Endpoint ${{endpoint}}: ${{error.message}}`, 'error');
                }}
            }}
            
            resultsArea.innerHTML += '\\n=== Test terminé ===\\n';
        }}
        
        async function testConnectivity() {{
            log('Test de connectivité...', 'info');
            try {{
                const response = await fetch(`${{API_BASE}}/health`);
                if (response.ok) {{
                    document.getElementById('api-status').innerHTML = '<span class="status-ok">✅ API Active</span>';
                    log('API accessible', 'success');
                }} else {{
                    document.getElementById('api-status').innerHTML = '<span class="status-error">❌ API Erreur</span>';
                    log(`API erreur: ${{response.status}}`, 'error');
                }}
            }} catch (error) {{
                document.getElementById('api-status').innerHTML = '<span class="status-error">❌ API Inaccessible</span>';
                log(`API inaccessible: ${{error.message}}`, 'error');
            }}
        }}
        
        async function refreshMonitoring() {{
            const monitoringArea = document.getElementById('monitoring-data');
            monitoringArea.innerHTML = 'Actualisation...\\n';
            
            try {{
                const [gpu, nodes, telemetry] = await Promise.all([
                    fetch(`${{API_BASE}}/api/gpu/info`).then(r => r.json()),
                    fetch(`${{API_BASE}}/api/nodes`).then(r => r.json()),
                    fetch(`${{API_BASE}}/api/telemetry.bin`).then(r => r.json())
                ]);
                
                monitoringArea.innerHTML = `GPU Info:\\n${{JSON.stringify(gpu, null, 2)}}\\n\\n`;
                monitoringArea.innerHTML += `Nodes:\\n${{JSON.stringify(nodes, null, 2)}}\\n\\n`;
                monitoringArea.innerHTML += `Télémétrie:\\n${{JSON.stringify(telemetry, null, 2)}}\\n`;
                
                log('Monitoring actualisé', 'success');
            }} catch (error) {{
                monitoringArea.innerHTML = `Erreur: ${{error.message}}`;
                log(`Erreur monitoring: ${{error.message}}`, 'error');
            }}
        }}
        
        function clearLog() {{
            document.getElementById('live-log').innerHTML = '';
            document.getElementById('api-results').innerHTML = 'Log effacé...\\n';
            logCounter = 0;
            log('Log effacé', 'info');
        }}
        
        async function testNetworkLatency() {{
            log('Test de latence réseau...', 'info');
            const latencies = [];
            
            for (let i = 0; i < 5; i++) {{
                try {{
                    const start = performance.now();
                    await fetch(`${{API_BASE}}/health`);
                    const end = performance.now();
                    latencies.push(end - start);
                }} catch (error) {{
                    log(`Ping ${{i+1}} échoué: ${{error.message}}`, 'error');
                }}
            }}
            
            if (latencies.length > 0) {{
                const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
                const min = Math.min(...latencies);
                const max = Math.max(...latencies);
                log(`Latence - Min: ${{min.toFixed(1)}}ms, Max: ${{max.toFixed(1)}}ms, Moy: ${{avg.toFixed(1)}}ms`, 'success');
            }}
        }}
        
        function downloadDebugInfo() {{
            const debugInfo = {{
                timestamp: new Date().toISOString(),
                userAgent: navigator.userAgent,
                url: window.location.href,
                apiBase: API_BASE,
                logContent: document.getElementById('live-log').innerHTML,
                apiResults: document.getElementById('api-results').innerHTML
            }};
            
            const blob = new Blob([JSON.stringify(debugInfo, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'vramancer-debug-info.json';
            a.click();
            URL.revokeObjectURL(url);
            
            log('Debug info téléchargé', 'success');
        }}
        
        // Initialisation
        window.addEventListener('load', () => {{
            log('Dashboard Debug initialisé', 'success');
            testConnectivity();
            refreshMonitoring();
            
            // Auto-refresh toutes les 30 secondes
            setInterval(() => {{
                testConnectivity();
            }}, 30000);
        }});
    </script>
</body>
</html>
            '''
        
        @app.route('/debug/info')
        def debug_info_endpoint():
            return jsonify({
                "python_version": sys.version,
                "flask_version": "installed",
                "api_base": os.environ.get('VRM_API_BASE', 'http://localhost:5030'),
                "working_directory": os.getcwd(),
                "environment_vars": {k: v for k, v in os.environ.items() if 'VRM' in k},
                "requests_count": debug_info["requests_count"],
                "last_error": debug_info["last_error"]
            })
        
        @app.before_request
        def before_request():
            debug_info["requests_count"] += 1
        
        return app
        
    except Exception as e:
        print_status(f"Erreur création serveur debug: {e}", "ERROR")
        traceback.print_exc()
        return None

def main():
    print_section("VRAMANCER WEB DEBUG - DIAGNOSTIC COMPLET")
    
    # Configuration environnement
    os.environ['VRM_API_BASE'] = 'http://localhost:5030'
    os.environ['VRM_API_PORT'] = '5030'
    
    print_status("Variables d'environnement configurées", "SUCCESS")
    print_status(f"VRM_API_BASE = {os.environ['VRM_API_BASE']}", "INFO")
    
    # Test et installation des dépendances
    print_section("VÉRIFICATION DÉPENDANCES")
    
    deps_ok = True
    deps_ok &= install_package("flask")
    deps_ok &= install_package("requests") 
    
    if not deps_ok:
        print_status("Certaines dépendances manquent, mais on continue...", "WARNING")
    
    # Test de l'API
    api_ok = test_api_connection()
    
    # Test Flask
    app = test_flask_setup()
    if not app:
        print_status("Impossible de configurer Flask", "ERROR")
        return
    
    # Création du serveur de debug
    debug_app = create_debug_web_server()
    if not debug_app:
        print_status("Impossible de créer le serveur debug", "ERROR")
        return
    
    # Démarrage du serveur
    print_section("DÉMARRAGE SERVEUR DEBUG")
    
    def open_browser():
        time.sleep(3)
        webbrowser.open('http://localhost:8080')
        print_status("Navigateur ouvert sur http://localhost:8080", "SUCCESS")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    print_status("Serveur debug démarré", "SUCCESS")
    print_status("URL: http://localhost:8080", "INFO")
    print_status("Diagnostic complet disponible dans l'interface web", "INFO")
    
    if not api_ok:
        print_status("⚠️ API non accessible - Démarrez d'abord: python start_api.py", "WARNING")
    
    print_status("Appuyez sur Ctrl+C pour arrêter", "INFO")
    
    try:
        debug_app.run(host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        print_status("\nServeur debug arrêté", "INFO")
    except Exception as e:
        print_status(f"Erreur serveur: {e}", "ERROR")
        traceback.print_exc()

if __name__ == "__main__":
    main()