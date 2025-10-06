#!/usr/bin/env python3
"""
API VRAMancer - Version ultra simple
"""
import os
import sys

# Définir les variables d'environnement
os.environ['VRM_API_BASE'] = 'http://localhost:5030'
os.environ['VRM_API_PORT'] = '5030'

print("🚀 Démarrage API VRAMancer simple...")
print(f"VRM_API_BASE = {os.environ['VRM_API_BASE']}")

try:
    from flask import Flask, jsonify
    print("✅ Flask disponible")
except ImportError:
    print("❌ Flask non disponible, installation...")
    os.system("python -m pip install flask")
    from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "vramancer"})

@app.route('/api/status')
def status():
    return jsonify({
        "backend": "running", 
        "version": "1.0",
        "api_base": os.environ.get('VRM_API_BASE', 'http://localhost:5030')
    })

@app.route('/api/gpu/info')
def gpu_info():
    return jsonify({
        "gpus": [{"id": 0, "name": "Default GPU", "memory": "Detecting..."}],
        "message": "API fonctionnelle - GPU detection en cours"
    })

if __name__ == '__main__':
    print("✅ API démarrée sur http://localhost:5030")
    print("   • Health: http://localhost:5030/health")
    print("   • Status: http://localhost:5030/api/status") 
    print("   • GPU: http://localhost:5030/api/gpu/info")
    print("\n🎮 Vous pouvez maintenant lancer les interfaces!")
    
    try:
        app.run(host='0.0.0.0', port=5030, debug=False)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        input("Appuyez sur Entrée...")