#!/usr/bin/env python3
"""
Debug Web Simplifié - Pour diagnostiquer problème interface web
"""

import os
import sys
import webbrowser
import threading
import time
from flask import Flask, render_template_string, jsonify

app = Flask(__name__)

# Template HTML simplifié
SIMPLE_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>VRAMancer Debug Simple</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            background: #1a1a1a; 
            color: #fff; 
            margin: 20px; 
        }
        .container { max-width: 800px; margin: 0 auto; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .ok { background: #2d5a2d; }
        .error { background: #5a2d2d; }
        .warning { background: #5a5a2d; }
        button { 
            padding: 10px 20px; 
            margin: 5px; 
            background: #4a4a4a; 
            color: white; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
        }
        button:hover { background: #6a6a6a; }
        #log { 
            background: #2a2a2a; 
            padding: 10px; 
            height: 300px; 
            overflow-y: scroll; 
            font-family: monospace; 
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 VRAMancer Debug Simple</h1>
        
        <div class="status" id="api-status">
            🔄 Test initial de l'API...
        </div>
        
        <div>
            <button onclick="testAPI()">Test API Manuel</button>
            <button onclick="clearLog()">Effacer Log</button>
            <button onclick="testAllEndpoints()">Test Tous Endpoints</button>
        </div>
        
        <h3>📋 Log en Temps Réel:</h3>
        <div id="log"></div>
    </div>

    <script>
        const API_BASE = 'http://localhost:5030';
        
        function log(message) {
            const logDiv = document.getElementById('log');
            const timestamp = new Date().toLocaleTimeString();
            logDiv.textContent += `[${timestamp}] ${message}\\n`;
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        function clearLog() {
            document.getElementById('log').textContent = '';
            log('Log effacé');
        }
        
        async function testAPI() {
            log('Test API manuel démarré...');
            const statusDiv = document.getElementById('api-status');
            
            try {
                log(`Tentative connexion: ${API_BASE}/health`);
                
                const response = await fetch(`${API_BASE}/health`, {
                    method: 'GET',
                    mode: 'cors'
                });
                
                log(`Réponse HTTP: ${response.status}`);
                
                if (response.ok) {
                    const data = await response.json();
                    log(`Données reçues: ${JSON.stringify(data)}`);
                    statusDiv.className = 'status ok';
                    statusDiv.textContent = '✅ API Active et Accessible';
                } else {
                    log(`Erreur HTTP: ${response.status} - ${response.statusText}`);
                    statusDiv.className = 'status error';
                    statusDiv.textContent = `❌ API Erreur: ${response.status}`;
                }
            } catch (error) {
                log(`Erreur catch: ${error.name} - ${error.message}`);
                statusDiv.className = 'status error';
                statusDiv.textContent = '❌ API Inaccessible';
            }
        }
        
        async function testAllEndpoints() {
            log('Test de tous les endpoints...');
            const endpoints = ['/health', '/api/status', '/api/gpu/info', '/api/nodes'];
            
            for (const endpoint of endpoints) {
                try {
                    log(`Test ${endpoint}...`);
                    const response = await fetch(`${API_BASE}${endpoint}`);
                    log(`${endpoint}: ${response.status} ${response.ok ? 'OK' : 'Erreur'}`);
                } catch (error) {
                    log(`${endpoint}: Erreur - ${error.message}`);
                }
            }
        }
        
        // Test automatique au chargement
        window.addEventListener('load', () => {
            log('Interface debug simple chargée');
            log('Test automatique API...');
            testAPI();
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return SIMPLE_HTML

@app.route('/test')
def test_endpoint():
    return jsonify({
        "message": "Debug web fonctionne",
        "status": "ok",
        "timestamp": time.time()
    })

def open_browser():
    time.sleep(1)
    webbrowser.open('http://localhost:8080')

if __name__ == '__main__':
    print("=" * 60)
    print("  VRAMANCER DEBUG WEB SIMPLIFIÉ")
    print("=" * 60)
    print()
    print("🔧 Interface de debug simplifiée")
    print("📱 URL: http://localhost:8080")
    print("🔍 Cette version teste uniquement la connectivité API")
    print()
    print("Ctrl+C pour arrêter")
    print()
    
    # Ouverture automatique du navigateur
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        app.run(host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        print("\\nDebug web simplifié arrêté")