#!/usr/bin/env python3
"""
Debug Web VRAMancer - Version Ultra Corrigée
Interface web avec JavaScript complètement fonctionnel
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
    """Crée l'application Flask de debug ultra corrigée"""
    try:
        from flask import Flask, jsonify, render_template_string
        
        app = Flask(__name__)
        
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
    <title>VRAMancer Debug Dashboard Ultra</title>
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
            transform: translateY(-1px);
        }
        .btn:active {
            transform: translateY(0px);
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
            border: 1px solid #444;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #444;
        }
        .metric:last-child { border-bottom: none; }
        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 VRAMancer - Debug Dashboard Ultra ⚡</h1>
            <p>Diagnostic complet avec JavaScript ULTRA corrigé</p>
        </div>
        
        <div class="debug-section">
            <h3>🌐 Configuration Serveur</h3>
            <div class="metric">
                <span>Serveur Web:</span>
                <span class="status-ok">✅ Actif sur port 8080</span>
            </div>
            <div class="metric">
                <span>API Backend:</span>
                <span id="api-status" class="pulse">🔄 Vérification...</span>
            </div>
            <div class="metric">
                <span>API Base URL:</span>
                <span>{{ api_base }}</span>
            </div>
            <div class="metric">
                <span>Status JavaScript:</span>
                <span id="js-status" class="status-warning">⚠️ Chargement...</span>
            </div>
        </div>
        
        <div class="debug-section">
            <h3>🧪 Tests API</h3>
            <button class="btn" onclick="testAllEndpoints()">🔍 Test Tous les Endpoints</button>
            <button class="btn" onclick="testConnectivityManual()">⚡ Test Connectivité</button>
            <button class="btn" onclick="testSingleEndpoint()">🎯 Test Health Seul</button>
            <button class="btn" onclick="clearLog()">🧹 Effacer Log</button>
            <div id="api-results" class="log-area">Prêt pour les tests API...</div>
        </div>
        
        <div class="debug-section">
            <h3>📊 Monitoring Live</h3>
            <button class="btn" onclick="refreshMonitoring()">🔄 Actualiser</button>
            <button class="btn" onclick="toggleAutoRefresh()">⏰ Auto-refresh</button>
            <button class="btn" onclick="debugConnectivity()">🔧 Debug Détaillé</button>
            <div id="live-log" class="log-area">Initialisation du monitoring...</div>
        </div>
        
        <div class="debug-section">
            <h3>🛠️ Debug Console</h3>
            <div id="debug-console" class="log-area">Console debug prête...</div>
        </div>
    </div>

    <script>
        const API_BASE = '{{ api_base }}';
        let autoRefreshEnabled = false;
        let autoRefreshInterval = null;
        
        // Fonctions utilitaires
        function debugLog(message, area = 'debug-console') {
            const debugArea = document.getElementById(area);
            const timestamp = new Date().toLocaleTimeString();
            debugArea.textContent += `[${timestamp}] ${message}\\n`;
            debugArea.scrollTop = debugArea.scrollHeight;
            console.log(`DEBUG: ${message}`);
        }
        
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
            
            // Aussi dans debug console
            debugLog(`LOG(${type}): ${message}`, 'debug-console');
        }
        
        // Fonction de timeout pour fetch
        function fetchWithTimeout(url, options = {}, timeout = 5000) {
            return Promise.race([
                fetch(url, options),
                new Promise((_, reject) =>
                    setTimeout(() => reject(new Error('Timeout')), timeout)
                )
            ]);
        }
        
        // Test connectivité manuel
        async function testConnectivityManual() {
            debugLog('🔧 Test connectivité MANUEL démarré');
            log('Test de connectivité API manuel...', 'info');
            
            const apiStatusElement = document.getElementById('api-status');
            apiStatusElement.innerHTML = '<span class="status-warning pulse">🔄 Test en cours...</span>';
            
            try {
                debugLog(`Tentative connexion: ${API_BASE}/health`);
                
                const response = await fetchWithTimeout(`${API_BASE}/health`, {
                    method: 'GET',
                    headers: {
                        'Accept': 'application/json',
                        'Content-Type': 'application/json'
                    }
                }, 5000);
                
                debugLog(`Response status: ${response.status}`);
                
                if (response.ok) {
                    const data = await response.json();
                    apiStatusElement.innerHTML = '<span class="status-ok">✅ API Connectée</span>';
                    log(`API Health OK: ${JSON.stringify(data)}`, 'success');
                    debugLog(`API Data: ${JSON.stringify(data)}`);
                    return true;
                } else {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
            } catch (error) {
                debugLog(`Erreur connectivité: ${error.message}`);
                console.error('Erreur connectivité:', error);
                apiStatusElement.innerHTML = '<span class="status-error">❌ API Déconnectée</span>';
                log(`Echec connexion API: ${error.message}`, 'error');
                return false;
            }
        }
        
        // Test endpoint unique
        async function testSingleEndpoint() {
            debugLog('Test endpoint /health uniquement');
            log('Test endpoint Health uniquement...', 'info');
            
            const resultsArea = document.getElementById('api-results');
            resultsArea.textContent = 'Test Health endpoint...\\n';
            
            try {
                const response = await fetchWithTimeout(`${API_BASE}/health`, {}, 3000);
                
                if (response.ok) {
                    const data = await response.json();
                    resultsArea.textContent += `✅ /health: ${JSON.stringify(data)}\\n`;
                    log('/health: SUCCESS', 'success');
                } else {
                    resultsArea.textContent += `❌ /health: Status ${response.status}\\n`;
                    log(`/health: Error ${response.status}`, 'error');
                }
            } catch (error) {
                resultsArea.textContent += `❌ /health: ${error.message}\\n`;
                log(`/health: Exception ${error.message}`, 'error');
            }
        }
        
        // Test tous endpoints
        async function testAllEndpoints() {
            debugLog('Démarrage test complet endpoints');
            log('Démarrage test complet des endpoints...', 'info');
            const resultsArea = document.getElementById('api-results');
            resultsArea.textContent = '';
            
            const endpoints = ['/health', '/api/status', '/api/gpu/info', '/api/nodes', '/api/telemetry.bin'];
            
            for (const endpoint of endpoints) {
                try {
                    log(`Test ${endpoint}...`, 'info');
                    debugLog(`Testing endpoint: ${endpoint}`);
                    
                    const response = await fetchWithTimeout(`${API_BASE}${endpoint}`, {}, 5000);
                    
                    if (response.ok) {
                        const contentType = response.headers.get('content-type');
                        let data;
                        
                        if (contentType && contentType.includes('application/json')) {
                            data = await response.json();
                            data = JSON.stringify(data).substring(0, 100);
                        } else {
                            data = await response.text();
                            data = data.substring(0, 100);
                        }
                        
                        resultsArea.textContent += `✅ ${endpoint}: OK (${data}...)\\n`;
                        log(`${endpoint}: SUCCESS`, 'success');
                    } else {
                        resultsArea.textContent += `❌ ${endpoint}: Status ${response.status}\\n`;
                        log(`${endpoint}: Error ${response.status}`, 'error');
                    }
                } catch (error) {
                    resultsArea.textContent += `❌ ${endpoint}: ${error.message}\\n`;
                    log(`${endpoint}: Exception ${error.message}`, 'error');
                    debugLog(`Endpoint ${endpoint} error: ${error.message}`);
                }
                
                // Petit délai entre tests
                await new Promise(resolve => setTimeout(resolve, 500));
            }
            
            log('Test complet terminé', 'success');
            debugLog('Test complet endpoints terminé');
        }
        
        // Debug détaillé
        async function debugConnectivity() {
            debugLog('=== DEBUG DÉTAILLÉ CONNECTIVITÉ ===');
            log('Debug connectivité détaillé...', 'info');
            
            // Test 1: Navigateur info
            debugLog(`User Agent: ${navigator.userAgent}`);
            debugLog(`URL actuelle: ${window.location.href}`);
            debugLog(`API_BASE configuré: ${API_BASE}`);
            
            // Test 2: Réseau
            debugLog('Test réseau basique...');
            try {
                const start = performance.now();
                const response = await fetch(`${API_BASE}/health`);
                const end = performance.now();
                debugLog(`Latence: ${(end - start).toFixed(2)}ms`);
                debugLog(`Status: ${response.status}`);
                debugLog(`Headers: ${JSON.stringify([...response.headers])}`);
            } catch (error) {
                debugLog(`Erreur réseau: ${error.message}`);
            }
            
            log('Debug détaillé terminé', 'success');
        }
        
        function clearLog() {
            document.getElementById('live-log').textContent = '';
            document.getElementById('api-results').textContent = 'Log effacé...';
            document.getElementById('debug-console').textContent = 'Console debug effacée...';
            log('Logs effacés', 'info');
        }
        
        function refreshMonitoring() {
            log('Actualisation monitoring...', 'info');
            testConnectivityManual();
        }
        
        function toggleAutoRefresh() {
            if (autoRefreshEnabled) {
                clearInterval(autoRefreshInterval);
                autoRefreshEnabled = false;
                log('Auto-refresh désactivé', 'warning');
            } else {
                autoRefreshInterval = setInterval(testConnectivityManual, 30000);
                autoRefreshEnabled = true;
                log('Auto-refresh activé (30s)', 'success');
            }
        }
        
        // Initialisation
        window.addEventListener('load', () => {
            debugLog('=== INITIALISATION DASHBOARD ===');
            document.getElementById('js-status').innerHTML = '<span class="status-ok">✅ JavaScript OK</span>';
            log('Dashboard Debug Ultra initialisé', 'success');
            debugLog('Démarrage test connectivité automatique...');
            testConnectivityManual();
        });
        
        // Gestion erreurs globales
        window.onerror = function(msg, url, lineNo, columnNo, error) {
            debugLog(`ERREUR JS: ${msg} (ligne ${lineNo})`);
            log(`Erreur JavaScript: ${msg}`, 'error');
        };
    </script>
</body>
</html>
            '''
            
            return render_template_string(html_template, api_base=api_base)
        
        @app.route('/api/debug/test')
        def test_endpoint():
            return jsonify({
                "status": "ok",
                "message": "Debug Web Ultra endpoint fonctionnel",
                "timestamp": datetime.now().isoformat()
            })
        
        return app
        
    except Exception as e:
        print_status(f"Erreur création serveur debug: {e}", "ERROR")
        traceback.print_exc()
        return None

def main():
    print_section("VRAMANCER DEBUG WEB ULTRA CORRIGÉ")
    
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
    
    print_status("Serveur Debug Ultra créé", "SUCCESS")
    
    # Démarrage
    print_section("DÉMARRAGE SERVEUR ULTRA CORRIGÉ")
    print_status("Serveur démarré sur http://localhost:8080")
    print_status("Interface JavaScript ULTRA corrigée avec debug détaillé")
    print_status("Boutons de test multiples disponibles")
    print_status("Appuyez sur Ctrl+C pour arrêter")
    
    try:
        webbrowser.open('http://localhost:8080')
        app.run(host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        print_status("Serveur arrêté", "INFO")

if __name__ == "__main__":
    main()