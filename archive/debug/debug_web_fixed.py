#!/usr/bin/env python3
"""
Debug Web VRAMancer - Version corrigée
Interface web pour diagnostic complet avec JavaScript fixé
"""

import os
import sys
import time
import json
import traceback
import subprocess
import webbrowser
from datetime import datetime

def print_section(title):
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_status(message, status="INFO"):
    symbols = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
    print(f"{symbols.get(status, 'ℹ️')} {message}")

def test_api_connectivity():
    """Test rapide de connectivité API"""
    try:
        import requests
        response = requests.get('http://localhost:5030/health', timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, f"Status code: {response.status_code}"
    except Exception as e:
        return False, str(e)

def create_debug_app():
    """Crée l'application Flask de debug corrigée"""
    try:
        from flask import Flask, jsonify, render_template_string
        from flask_cors import CORS, cross_origin
        
        app = Flask(__name__)
        CORS(app, resources={r"/*": {"origins": "*"}})
        
        debug_info = {"requests_count": 0, "last_error": None}
        
        @app.route('/')
        def debug_dashboard():
            api_base = os.environ.get('VRM_API_BASE', 'http://localhost:5030')
            
            html_template = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VRAMancer Debug Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .debug-section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        .btn {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            font-size: 14px;
        }
        .btn:hover {
            background: linear-gradient(45deg, #45a049, #4CAF50);
        }
        .status-ok { color: #4CAF50; }
        .status-error { color: #f44336; }
        .status-warning { color: #ff9800; }
        .log-area {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 5px;
            padding: 15px;
            font-family: monospace;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #444;
        }
        .metric:last-child { border-bottom: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 VRAMancer - Debug Dashboard Web (FIXÉ)</h1>
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
                <span>API Base URL:</span>
                <span>{{ api_base }}</span>
            </div>
        </div>
        
        <div class="debug-section">
            <h3>🧪 Tests API</h3>
            <button class="btn" onclick="testAllEndpoints()">Test Tous les Endpoints</button>
            <button class="btn" onclick="testConnectivityNow()">Test Connectivité</button>
            <button class="btn" onclick="clearLog()">Effacer Log</button>
            <div id="api-results" class="log-area">Cliquez sur "Test Tous les Endpoints" pour commencer...</div>
        </div>
        
        <div class="debug-section">
            <h3>📊 Monitoring Live</h3>
            <button class="btn" onclick="refreshMonitoring()">Actualiser</button>
            <button class="btn" onclick="toggleAutoRefresh()">Auto-refresh</button>
            <div id="live-log" class="log-area">Initialisation...</div>
        </div>
    </div>

    <script>
        const API_BASE = '{{ api_base }}';
        let autoRefreshEnabled = false;
        let autoRefreshInterval = null;
        
        function log(message, type = 'info') {
            const logArea = document.getElementById('live-log');
            const timestamp = new Date().toLocaleTimeString();
            const symbols = {
                'info': 'ℹ️',
                'success': '✅',
                'error': '❌',
                'warning': '⚠️'
            };
            
            const logEntry = `[${timestamp}] ${symbols[type] || 'ℹ️'} ${message}\\n`;
            logArea.textContent += logEntry;
            logArea.scrollTop = logArea.scrollHeight;
        }
        
        async function testConnectivityNow() {
            console.log('🔧 Test connectivité démarré');
            log('Test de connectivité API...', 'info');
            
            const apiStatusElement = document.getElementById('api-status');
            
            try {
                const response = await fetch(`${API_BASE}/health`, {
                    method: 'GET',
                    timeout: 5000
                });
                
                if (response.ok) {
                    const data = await response.json();
                    apiStatusElement.innerHTML = '<span class="status-ok">✅ API Connectée</span>';
                    log(`API Health OK: ${JSON.stringify(data)}`, 'success');
                    return true;
                } else {
                    throw new Error(`Status: ${response.status}`);
                }
            } catch (error) {
                console.error('Erreur connectivité:', error);
                apiStatusElement.innerHTML = '<span class="status-error">❌ API Déconnectée</span>';
                log(`Echec connexion API: ${error.message}`, 'error');
                return false;
            }
        }
        
        async function testAllEndpoints() {
            log('Démarrage test complet des endpoints...', 'info');
            const resultsArea = document.getElementById('api-results');
            resultsArea.textContent = '';
            
            const endpoints = ['/health', '/api/status', '/api/gpu/info', '/api/nodes', '/api/telemetry.bin'];
            
            for (const endpoint of endpoints) {
                try {
                    log(`Test ${endpoint}...`, 'info');
                    const response = await fetch(`${API_BASE}${endpoint}`);
                    
                    if (response.ok) {
                        const data = await response.text();
                        resultsArea.textContent += `✅ ${endpoint}: OK\\n`;
                        log(`${endpoint}: SUCCESS`, 'success');
                    } else {
                        resultsArea.textContent += `❌ ${endpoint}: Status ${response.status}\\n`;
                        log(`${endpoint}: Error ${response.status}`, 'error');
                    }
                } catch (error) {
                    resultsArea.textContent += `❌ ${endpoint}: ${error.message}\\n`;
                    log(`${endpoint}: Exception ${error.message}`, 'error');
                }
                
                // Petit délai entre tests
                await new Promise(resolve => setTimeout(resolve, 500));
            }
            
            log('Test complet terminé', 'success');
        }
        
        function clearLog() {
            document.getElementById('live-log').textContent = '';
            document.getElementById('api-results').textContent = 'Log effacé...';
            log('Log effacé', 'info');
        }
        
        function refreshMonitoring() {
            log('Actualisation monitoring...', 'info');
            testConnectivityNow();
        }
        
        function toggleAutoRefresh() {
            if (autoRefreshEnabled) {
                clearInterval(autoRefreshInterval);
                autoRefreshEnabled = false;
                log('Auto-refresh désactivé', 'warning');
            } else {
                autoRefreshInterval = setInterval(testConnectivityNow, 30000);
                autoRefreshEnabled = true;
                log('Auto-refresh activé (30s)', 'success');
            }
        }
        
        // Initialisation au chargement
        window.addEventListener('load', () => {
            log('Dashboard Debug Web initialisé', 'success');
            testConnectivityNow();
        });
    </script>
</body>
</html>
            '''
            
            return render_template_string(html_template, api_base=api_base)
        
        @app.route('/api/debug/test')
        @cross_origin()
        def test_endpoint():
            return jsonify({
                "status": "ok",
                "message": "Debug Web endpoint fonctionnel",
                "timestamp": datetime.now().isoformat()
            })
        
        return app
        
    except Exception as e:
        print_status(f"Erreur création serveur debug: {e}", "ERROR")
        traceback.print_exc()
        return None

def main():
    print_section("VRAMANCER DEBUG WEB CORRIGÉ")
    
    # Configuration
    os.environ['VRM_API_BASE'] = 'http://localhost:5030'
    print_status("Variables d'environnement configurées")
    
    # Test API
    connected, result = test_api_connectivity()
    if connected:
        print_status(f"API connectée: {result}", "SUCCESS")
    else:
        print_status(f"API non connectée: {result}", "WARNING")
    
    # Création serveur
    app = create_debug_app()
    if not app:
        print_status("Impossible de créer le serveur", "ERROR")
        return
    
    print_status("Serveur Debug corrigé créé", "SUCCESS")
    
    # Démarrage
    print_section("DÉMARRAGE SERVEUR CORRIGÉ")
    print_status("Serveur démarré sur http://localhost:8080")
    print_status("Interface JavaScript corrigée")
    print_status("Appuyez sur Ctrl+C pour arrêter")
    
    try:
        webbrowser.open('http://localhost:8080')
        app.run(host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        print_status("Serveur arrêté", "INFO")

if __name__ == "__main__":
    main()