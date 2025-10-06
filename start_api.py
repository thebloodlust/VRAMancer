#!/usr/bin/env python3
"""
VRAMancer API Launcher - Démarrage garanti de l'API
"""

import os
import sys
import time
import subprocess
import requests
import threading
from pathlib import Path

def print_status(message, status="INFO"):
    colors = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
    print(f"{colors.get(status, 'ℹ️')} {message}")

def check_api_health(port=5030, timeout=2):
    """Vérification santé de l'API"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def install_dependencies():
    """Installation des dépendances critiques"""
    print_status("Installation des dépendances...", "INFO")
    deps = ["flask", "flask-cors", "requests", "psutil", "pyyaml"]
    
    for dep in deps:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep, 
                "--quiet", "--disable-pip-version-check"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            print_status(f"Échec installation {dep}", "WARNING")

def start_minimal_api(port=5030):
    """Démarrage d'une API minimale de fallback"""
    try:
        from flask import Flask, jsonify
        from flask_cors import CORS
        
        app = Flask(__name__)
        CORS(app)
        
        @app.route('/health')
        def health():
            return jsonify({"status": "ok", "service": "vramancer-api"})
        
        @app.route('/api/status')
        def api_status():
            return jsonify({
                "backend": "running",
                "version": "1.0.0",
                "mode": "minimal"
            })
        
        @app.route('/api/gpu/info')
        def gpu_info():
            return jsonify({
                "gpus": [{"id": 0, "name": "Auto-detected", "memory": "N/A"}],
                "total_memory": "Detecting..."
            })
        
        print_status(f"API minimale démarrée sur port {port}", "SUCCESS")
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
        
    except Exception as e:
        print_status(f"Erreur API minimale: {e}", "ERROR")
        return False

def start_full_api():
    """Tentative de démarrage de l'API complète"""
    
    # Méthode 1: Module core.api.unified_api
    try:
        print_status("Tentative: python -m core.api.unified_api", "INFO")
        from core.api.unified_api import app
        print_status("API complète trouvée, démarrage...", "SUCCESS")
        app.run(host='0.0.0.0', port=5030, debug=False, use_reloader=False)
        return True
    except ImportError:
        print_status("Module core.api.unified_api non trouvé", "WARNING")
    except Exception as e:
        print_status(f"Erreur API complète: {e}", "WARNING")
    
    # Méthode 2: Import direct
    try:
        sys.path.append('.')
        from core.api import unified_api
        unified_api.app.run(host='0.0.0.0', port=5030, debug=False)
        return True
    except:
        pass
    
    # Méthode 3: API minimale
    print_status("Démarrage API minimale de fallback...", "INFO")
    return start_minimal_api()

def main():
    print("=" * 60)
    print("🚀 VRAMancer API Launcher")
    print("=" * 60)
    
    # Installation dépendances
    install_dependencies()
    
    # Vérification si API déjà active
    if check_api_health():
        print_status("API déjà active sur localhost:5030", "SUCCESS")
        return
    
    # Variables d'environnement
    os.environ['VRM_API_BASE'] = 'http://localhost:5030'
    os.environ['VRM_API_PORT'] = '5030'
    
    print_status("Démarrage de l'API VRAMancer...", "INFO")
    
    # Démarrage en thread séparé pour permettre les vérifications
    api_thread = threading.Thread(target=start_full_api, daemon=True)
    api_thread.start()
    
    # Attente et vérification
    for i in range(10):
        time.sleep(1)
        if check_api_health():
            print_status("✅ API active sur http://localhost:5030", "SUCCESS")
            print_status("✅ Variables d'environnement configurées", "SUCCESS")
            print("\n🌐 Points d'accès API:")
            print("   • Health check: http://localhost:5030/health")
            print("   • Status: http://localhost:5030/api/status")
            print("   • GPU Info: http://localhost:5030/api/gpu/info")
            print("\n🎮 Vous pouvez maintenant lancer:")
            print("   • Dashboard Web: python dashboard/dashboard_web.py")
            print("   • Interface Qt: python dashboard/dashboard_qt.py")
            print("   • Systray: python systray_vramancer.py")
            
            # Maintenir l'API active
            try:
                while True:
                    time.sleep(30)
                    if not check_api_health():
                        print_status("API déconnectée, tentative de redémarrage...", "WARNING")
                        break
            except KeyboardInterrupt:
                print_status("\nArrêt de l'API", "INFO")
            return
    
    print_status("Échec démarrage API après 10 secondes", "ERROR")
    print("\n🔧 Solutions:")
    print("1. Vérifiez que le port 5030 n'est pas utilisé")
    print("2. Essayez: netstat -an | findstr :5030")
    print("3. Relancez en tant qu'administrateur")

if __name__ == "__main__":
    main()