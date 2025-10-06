#!/usr/bin/env python3
"""
Test du Dashboard Web - Diagnostic
"""

import os
import sys
import time
import webbrowser
import threading

print("🧪 Test Dashboard Web VRAMancer")
print("=" * 50)

# Variables d'environnement
os.environ['VRM_API_BASE'] = 'http://localhost:5030'

# Test des imports
print("1. Test des imports...")
try:
    from flask import Flask, render_template_string, jsonify
    print("   ✅ Flask OK")
except ImportError as e:
    print(f"   ❌ Flask manquant: {e}")
    print("   Installation automatique...")
    os.system("python -m pip install flask")
    from flask import Flask, render_template_string, jsonify

try:
    import requests
    print("   ✅ Requests OK")
except ImportError:
    print("   ❌ Requests manquant")
    os.system("python -m pip install requests")
    import requests

# Test de l'API
print("\n2. Test de connexion API...")
try:
    response = requests.get('http://localhost:5030/health', timeout=3)
    if response.status_code == 200:
        print("   ✅ API accessible")
        print(f"   Réponse: {response.json()}")
    else:
        print(f"   ⚠️ API répond avec code {response.status_code}")
except Exception as e:
    print(f"   ❌ API inaccessible: {e}")
    print("   Démarrez d'abord l'API avec: python start_api.py")

# Création d'un serveur web minimal
print("\n3. Création serveur web minimal...")

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>VRAMancer Test Web</title>
        <style>
            body { background: #1a1a1a; color: #fff; font-family: Arial; padding: 20px; }
            .card { background: #2d2d2d; padding: 20px; border-radius: 8px; margin: 10px 0; }
            .btn { background: #007acc; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        </style>
    </head>
    <body>
        <h1>🚀 VRAMancer - Test Dashboard Web</h1>
        
        <div class="card">
            <h3>✅ Serveur Web Fonctionnel</h3>
            <p>Ce test confirme que Flask fonctionne correctement.</p>
        </div>
        
        <div class="card">
            <h3>🔧 Test API</h3>
            <button class="btn" onclick="testAPI()">Test API Health</button>
            <div id="api-result"></div>
        </div>
        
        <div class="card">
            <h3>🎮 Actions</h3>
            <button class="btn" onclick="window.open('http://localhost:5030/health', '_blank')">Ouvrir API Health</button>
            <button class="btn" onclick="window.open('http://localhost:5030/api/status', '_blank')">Ouvrir API Status</button>
        </div>
        
        <script>
            async function testAPI() {
                const result = document.getElementById('api-result');
                result.innerHTML = 'Test en cours...';
                
                try {
                    const response = await fetch('http://localhost:5030/health');
                    const data = await response.json();
                    result.innerHTML = `<p style="color: #28a745;">✅ API OK: ${JSON.stringify(data)}</p>`;
                } catch (error) {
                    result.innerHTML = `<p style="color: #dc3545;">❌ Erreur: ${error.message}</p>`;
                }
            }
            
            // Test automatique au chargement
            window.onload = () => {
                console.log('Dashboard Web Test chargé');
                testAPI();
            };
        </script>
    </body>
    </html>
    '''

def open_browser():
    time.sleep(2)
    webbrowser.open('http://localhost:8080')
    print("   🌐 Navigateur ouvert sur http://localhost:8080")

if __name__ == '__main__':
    print("\n4. Démarrage du serveur...")
    print("   Port: 8080")
    print("   URL: http://localhost:8080")
    
    # Ouvrir le navigateur automatiquement
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        print("   Serveur démarré - Appuyez sur Ctrl+C pour arrêter")
        app.run(host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        print("\n   Serveur arrêté")
    except Exception as e:
        print(f"\n   ❌ Erreur serveur: {e}")
        input("Appuyez sur Entrée...")