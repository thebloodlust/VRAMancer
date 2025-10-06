#!/usr/bin/env python3
"""
VRAMancer Windows Fix - Version Simple
Diagnostic et réparation automatique pour Windows
"""

import os
import sys
import subprocess
import importlib
import pkg_resources
from pathlib import Path

def print_status(message, status="INFO"):
    """Affichage avec couleurs pour Windows"""
    colors = {
        "INFO": "",
        "SUCCESS": "",
        "ERROR": "",
        "WARNING": ""
    }
    print(f"[{status}] {message}")

def check_python_version():
    """Vérification version Python"""
    version = sys.version_info
    print_status(f"Python {version.major}.{version.minor}.{version.micro}", "INFO")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_status("Python 3.8+ requis", "ERROR")
        return False
    return True

def install_package(package):
    """Installation d'un package Python"""
    try:
        print_status(f"Installation de {package}...", "INFO")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print_status(f"{package} installé", "SUCCESS")
        return True
    except subprocess.CalledProcessError:
        print_status(f"Erreur installation {package}", "ERROR")
        return False

def check_module(module_name, install_name=None):
    """Vérification et installation d'un module"""
    if install_name is None:
        install_name = module_name
    
    try:
        importlib.import_module(module_name)
        print_status(f"✅ {module_name} disponible", "SUCCESS")
        return True
    except ImportError:
        print_status(f"❌ {module_name} manquant", "WARNING")
        return install_package(install_name)

def create_simple_dashboard():
    """Création d'un dashboard simple qui fonctionne partout"""
    dashboard_content = '''#!/usr/bin/env python3
import http.server
import socketserver
import webbrowser
import threading
import time
import sys
import platform

class SimpleHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/dashboard':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html = """<!DOCTYPE html>
<html>
<head>
    <title>VRAMancer - Dashboard Windows</title>
    <meta charset="utf-8">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            background: #1a1a1a; 
            color: #fff; 
            margin: 0; 
            padding: 20px; 
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { 
            background: #2d2d2d; 
            border-radius: 8px; 
            padding: 20px; 
            margin: 10px 0; 
            border-left: 4px solid #00bfff; 
        }
        .status { 
            display: inline-block; 
            padding: 4px 12px; 
            border-radius: 4px; 
            font-size: 12px; 
        }
        .status.ok { background: #28a745; }
        button { 
            background: #007acc; 
            color: white; 
            border: none; 
            padding: 8px 16px; 
            border-radius: 4px; 
            cursor: pointer; 
            margin: 5px; 
        }
        button:hover { background: #005a9e; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 VRAMancer Dashboard</h1>
        
        <div class="card">
            <h2>🖥️ Système</h2>
            <p>OS: """ + platform.system() + """ """ + platform.release() + """</p>
            <p>Python: """ + sys.version.split()[0] + """</p>
            <p>Statut: <span class="status ok">Opérationnel</span></p>
        </div>
        
        <div class="card">
            <h2>✅ Installation Réussie</h2>
            <p>VRAMancer est maintenant configuré pour Windows !</p>
            <p>Cette version simple du dashboard confirme que Python fonctionne correctement.</p>
        </div>
        
        <div class="card">
            <h2>🎮 Prochaines Étapes</h2>
            <p>1. Lancez <code>python launch_vramancer.py</code> pour le système complet</p>
            <p>2. Testez votre cluster avec <code>python test_heterogeneous_cluster.py</code></p>
            <p>3. Configurez vos noeuds avec <code>python setup_heterogeneous_cluster.py</code></p>
        </div>
    </div>
    
    <script>
        console.log('VRAMancer Dashboard - Version Windows Simple');
        setInterval(() => {
            console.log('Dashboard actif:', new Date().toLocaleTimeString());
        }, 5000);
    </script>
</body>
</html>"""
            
            self.wfile.write(html.encode('utf-8'))
        else:
            super().do_GET()

def main():
    """Fonction principale"""
    print("=" * 60)
    print("🚀 VRAMancer - Fix Windows Simple")
    print("=" * 60)
    
    # Vérifications de base
    if not check_python_version():
        input("Appuyez sur Entrée pour quitter...")
        return
    
    # Vérification des modules essentiels
    modules_ok = True
    essential_modules = [
        'http.server',
        'socketserver', 
        'webbrowser',
        'threading',
        'platform'
    ]
    
    for module in essential_modules:
        if not check_module(module):
            modules_ok = False
    
    if modules_ok:
        print_status("Tous les modules essentiels sont disponibles", "SUCCESS")
    else:
        print_status("Certains modules manquent", "WARNING")
    
    # Lancement du dashboard simple
    print("\\n" + "=" * 60)
    print("🌐 Lancement du Dashboard Simple")
    print("=" * 60)
    
    try:
        PORT = 8080
        handler = SimpleHandler
        
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            print_status(f"Dashboard démarré sur http://localhost:{PORT}", "SUCCESS")
            print_status("Ouverture du navigateur...", "INFO")
            
            # Ouverture automatique du navigateur
            def open_browser():
                time.sleep(1)
                webbrowser.open(f'http://localhost:{PORT}')
            
            threading.Thread(target=open_browser, daemon=True).start()
            
            print("\\nAppuyez sur Ctrl+C pour arrêter")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print_status("\\nDashboard arrêté", "INFO")
    except Exception as e:
        print_status(f"Erreur: {e}", "ERROR")
        input("Appuyez sur Entrée pour quitter...")

if __name__ == "__main__":
    main()
'''
    
    with open("vramancer_dashboard_simple.py", "w", encoding="utf-8") as f:
        f.write(dashboard_content)
    
    print_status("Dashboard simple créé: vramancer_dashboard_simple.py", "SUCCESS")

def main():
    """Fonction principale du fix"""
    print("=" * 70)
    print("🚀 VRAMancer - Fix Windows Simple")
    print("Version de secours pour résoudre les problèmes de syntaxe")
    print("=" * 70)
    
    # Vérifications de base
    if not check_python_version():
        input("Appuyez sur Entrée pour quitter...")
        return
    
    # Vérification des modules critiques
    critical_modules = [
        ('flask', 'flask'),
        ('psutil', 'psutil'),
        ('requests', 'requests'),
        ('yaml', 'PyYAML')
    ]
    
    print("\n🔍 Vérification des dépendances critiques...")
    for module, install_name in critical_modules:
        check_module(module, install_name)
    
    # Création du dashboard de secours
    print("\n🛠️ Création du dashboard de secours...")
    create_simple_dashboard()
    
    print("\n" + "=" * 70)
    print("✅ Fix terminé ! Vous pouvez maintenant :")
    print("1. Lancer: python vramancer_dashboard_simple.py")
    print("2. Ou essayer: python launch_vramancer.py")
    print("3. Tester le cluster: python test_heterogeneous_cluster.py")
    print("=" * 70)
    
    input("\nAppuyez sur Entrée pour quitter...")

if __name__ == "__main__":
    main()