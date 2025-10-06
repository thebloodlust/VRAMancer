#!/usr/bin/env python3
"""
VRAMancer - Dashboard Web Simple
Fonctionne avec l'API minimale
"""

import os
import sys
import time
import webbrowser
import threading
from flask import Flask, render_template_string, jsonify
import requests

# Configuration
os.environ['VRM_API_BASE'] = 'http://localhost:5030'
API_BASE = os.environ.get('VRM_API_BASE', 'http://localhost:5030')
WEB_PORT = 8080

app = Flask(__name__)

# Template HTML intégré
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VRAMancer - Dashboard Web</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            color: #fff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            padding: 20px 0;
            border-bottom: 2px solid #00bfff;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(45deg, #00bfff, #0080ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #333;
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #00bfff;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        .status-indicator.online { background: #28a745; }
        .status-indicator.offline { background: #dc3545; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: #2d2d2d;
            border-radius: 15px;
            padding: 20px;
            border: 1px solid #444;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 191, 255, 0.2);
            border-color: #00bfff;
        }
        
        .card h3 {
            color: #00bfff;
            margin-bottom: 15px;
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #444;
        }
        
        .metric:last-child { border-bottom: none; }
        
        .metric-value {
            font-weight: bold;
            color: #00bfff;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        
        .btn {
            background: linear-gradient(45deg, #007acc, #00bfff);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }
        
        .btn:hover {
            background: linear-gradient(45deg, #005a9e, #007acc);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 123, 204, 0.4);
        }
        
        .btn.secondary {
            background: linear-gradient(45deg, #28a745, #20c997);
        }
        
        .btn.secondary:hover {
            background: linear-gradient(45deg, #1e7e34, #17a2b8);
        }
        
        #log {
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 15px;
            font-family: 'Consolas', monospace;
            font-size: 12px;
            max-height: 200px;
            overflow-y: auto;
            margin-top: 20px;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #888;
        }
        
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
            .status-bar { flex-direction: column; gap: 10px; }
            .controls { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 VRAMancer Dashboard</h1>
            <p>Cluster de calcul distribué - Interface Web</p>
        </div>
        
        <div class="status-bar">
            <div class="status-item">
                <div class="status-indicator" id="api-status"></div>
                <span id="api-text">Vérification API...</span>
            </div>
            <div class="status-item">
                <span>API: <strong>{{ api_base }}</strong></span>
            </div>
            <div class="status-item">
                <span id="last-update">Dernière mise à jour: --:--:--</span>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>🎮 Informations GPU</h3>
                <div id="gpu-info" class="loading">Chargement...</div>
                <div class="controls">
                    <button class="btn" onclick="refreshGPU()">🔄 Actualiser</button>
                </div>
            </div>
            
            <div class="card">
                <h3>🖥️ Noeuds du Cluster</h3>
                <div id="nodes-info" class="loading">Chargement...</div>
                <div class="controls">
                    <button class="btn" onclick="refreshNodes()">🔄 Actualiser</button>
                </div>
            </div>
            
            <div class="card">
                <h3>📊 Télémétrie</h3>
                <div id="telemetry-info" class="loading">Chargement...</div>
                <div class="controls">
                    <button class="btn" onclick="refreshTelemetry()">🔄 Actualiser</button>
                </div>
            </div>
            
            <div class="card">
                <h3>🔧 Actions</h3>
                <div class="controls">
                    <button class="btn" onclick="testAllEndpoints()">🧪 Test API</button>
                    <button class="btn secondary" onclick="window.open('{{ api_base }}/health', '_blank')">🌐 API Health</button>
                    <button class="btn secondary" onclick="window.open('{{ api_base }}/api/status', '_blank')">📋 Status</button>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>📝 Journal d'activité</h3>
            <div id="log"></div>
            <div class="controls">
                <button class="btn" onclick="clearLog()">🗑️ Effacer</button>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = '{{ api_base }}';
        let logCount = 0;
        
        function log(message, type = 'info') {
            const logDiv = document.getElementById('log');
            const timestamp = new Date().toLocaleTimeString();
            const colors = {
                'info': '#00bfff',
                'success': '#28a745', 
                'error': '#dc3545',
                'warning': '#ffc107'
            };
            
            logDiv.innerHTML += `<div style="color: ${colors[type] || '#fff'}">
                [${timestamp}] ${message}
            </div>`;
            logDiv.scrollTop = logDiv.scrollHeight;
            
            if (++logCount > 50) {
                const lines = logDiv.innerHTML.split('</div>');
                logDiv.innerHTML = lines.slice(-25).join('</div>');
                logCount = 25;
            }
        }
        
        function updateTimestamp() {
            document.getElementById('last-update').textContent = 
                'Dernière mise à jour: ' + new Date().toLocaleTimeString();
        }
        
        async function checkAPI() {
            try {
                const response = await fetch(`${API_BASE}/health`);
                if (response.ok) {
                    document.getElementById('api-status').className = 'status-indicator online';
                    document.getElementById('api-text').textContent = 'API Active';
                    return true;
                } else {
                    throw new Error(`HTTP ${response.status}`);
                }
            } catch (error) {
                document.getElementById('api-status').className = 'status-indicator offline';
                document.getElementById('api-text').textContent = 'API Inactive';
                log(`Erreur API: ${error.message}`, 'error');
                return false;
            }
        }
        
        async function refreshGPU() {
            try {
                const response = await fetch(`${API_BASE}/api/gpu/info`);
                const data = await response.json();
                
                let html = `<div class="metric">
                    <span>GPUs détectés:</span>
                    <span class="metric-value">${data.gpus ? data.gpus.length : 0}</span>
                </div>`;
                
                if (data.gpus) {
                    data.gpus.forEach((gpu, i) => {
                        html += `<div class="metric">
                            <span>GPU ${i}: ${gpu.name || 'Unknown'}</span>
                            <span class="metric-value">${gpu.memory || 'N/A'}</span>
                        </div>`;
                    });
                }
                
                html += `<div class="metric">
                    <span>Mémoire totale:</span>
                    <span class="metric-value">${data.total_memory || 'N/A'}</span>
                </div>`;
                
                document.getElementById('gpu-info').innerHTML = html;
                log('GPU info actualisée', 'success');
            } catch (error) {
                document.getElementById('gpu-info').innerHTML = 
                    `<div style="color: #dc3545;">Erreur: ${error.message}</div>`;
                log(`Erreur GPU: ${error.message}`, 'error');
            }
        }
        
        async function refreshNodes() {
            try {
                const response = await fetch(`${API_BASE}/api/nodes`);
                const data = await response.json();
                
                let html = `<div class="metric">
                    <span>Noeuds actifs:</span>
                    <span class="metric-value">${data.total_nodes || 0}</span>
                </div>`;
                
                if (data.nodes) {
                    data.nodes.forEach(node => {
                        html += `<div class="metric">
                            <span>${node.name || 'Unknown'}</span>
                            <span class="metric-value">${node.status || 'unknown'}</span>
                        </div>`;
                        html += `<div class="metric">
                            <span>GPU: ${node.gpu_count || 0}</span>
                            <span class="metric-value">${node.memory_used || 'N/A'}/${node.memory_total || 'N/A'}</span>
                        </div>`;
                    });
                }
                
                document.getElementById('nodes-info').innerHTML = html;
                log('Nodes info actualisées', 'success');
            } catch (error) {
                document.getElementById('nodes-info').innerHTML = 
                    `<div style="color: #dc3545;">Erreur: ${error.message}</div>`;
                log(`Erreur Nodes: ${error.message}`, 'error');
            }
        }
        
        async function refreshTelemetry() {
            try {
                const response = await fetch(`${API_BASE}/api/telemetry.bin`);
                const data = await response.json();
                
                let html = '';
                if (data.metrics) {
                    html += `<div class="metric">
                        <span>Usage GPU:</span>
                        <span class="metric-value">${data.metrics.gpu_usage || 0}%</span>
                    </div>`;
                    html += `<div class="metric">
                        <span>Usage Mémoire:</span>
                        <span class="metric-value">${data.metrics.memory_usage || 0}%</span>
                    </div>`;
                    html += `<div class="metric">
                        <span>Température:</span>
                        <span class="metric-value">${data.metrics.temperature || 0}°C</span>
                    </div>`;
                } else {
                    html = '<div>Aucune donnée de télémétrie</div>';
                }
                
                document.getElementById('telemetry-info').innerHTML = html;
                log('Télémétrie actualisée', 'success');
            } catch (error) {
                document.getElementById('telemetry-info').innerHTML = 
                    `<div style="color: #dc3545;">Erreur: ${error.message}</div>`;
                log(`Erreur Télémétrie: ${error.message}`, 'error');
            }
        }
        
        async function testAllEndpoints() {
            log('Test de tous les endpoints...', 'info');
            const endpoints = ['/health', '/api/status', '/api/gpu/info', '/api/nodes', '/api/telemetry.bin'];
            
            for (const endpoint of endpoints) {
                try {
                    const response = await fetch(`${API_BASE}${endpoint}`);
                    log(`✅ ${endpoint}: ${response.status}`, response.ok ? 'success' : 'error');
                } catch (error) {
                    log(`❌ ${endpoint}: ${error.message}`, 'error');
                }
            }
        }
        
        function clearLog() {
            document.getElementById('log').innerHTML = '';
            logCount = 0;
            log('Journal effacé', 'info');
        }
        
        // Initialisation
        window.addEventListener('load', async () => {
            log('Dashboard Web VRAMancer initialisé', 'success');
            
            if (await checkAPI()) {
                await Promise.all([refreshGPU(), refreshNodes(), refreshTelemetry()]);
            }
            
            updateTimestamp();
            
            // Actualisation automatique toutes les 30 secondes
            setInterval(async () => {
                if (await checkAPI()) {
                    await Promise.all([refreshGPU(), refreshNodes(), refreshTelemetry()]);
                    updateTimestamp();
                }
            }, 30000);
        });
    </script>
</body>
</html>
'''

@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE, api_base=API_BASE)

@app.route('/api/proxy/<path:endpoint>')
def proxy_api(endpoint):
    """Proxy pour éviter les problèmes CORS"""
    try:
        response = requests.get(f"{API_BASE}/api/{endpoint}")
        return response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def open_browser():
    """Ouvre automatiquement le navigateur"""
    time.sleep(2)
    webbrowser.open(f'http://localhost:{WEB_PORT}')

if __name__ == '__main__':
    print("🌐 Démarrage du Dashboard Web VRAMancer")
    print(f"   • Interface: http://localhost:{WEB_PORT}")
    print(f"   • API Backend: {API_BASE}")
    print("\n✨ Ouverture automatique du navigateur...")
    
    # Ouverture automatique du navigateur
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        app.run(host='0.0.0.0', port=WEB_PORT, debug=False)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        input("Appuyez sur Entrée...")