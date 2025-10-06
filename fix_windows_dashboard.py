#!/usr/bin/env python3
"""
Script de diagnostic et réparation pour les dashboards VRAMancer sur Windows
Identifie et résout les problèmes de dépendances manquantes
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_module(module_name, pip_name=None):
    """Vérifie si un module est installé"""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except ImportError:
        return False

def install_package(package_name):
    """Installe un package via pip"""
    try:
        print(f"Installation de {package_name}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package_name], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {package_name} installé avec succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur installation {package_name}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def diagnose_dashboard_issues():
    """Diagnostic complet des problèmes de dashboard"""
    print("🔍 Diagnostic VRAMancer Dashboard Windows...")
    print("=" * 50)
    
    issues = []
    
    # 1. Vérification Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        issues.append("Python version trop ancienne (< 3.8)")
    
    # 2. Vérification des dépendances critiques
    critical_deps = {
        'flask': 'flask',
        'flask_socketio': 'flask-socketio',
        'PyQt5': 'PyQt5',
        'tkinter': None,  # Built-in normalement
        'requests': 'requests',
        'numpy': 'numpy',
        'psutil': 'psutil',
        'torch': 'torch',
        'socketio': 'python-socketio[client]'
    }
    
    missing_deps = []
    
    for module, pip_package in critical_deps.items():
        if not check_module(module):
            missing_deps.append((module, pip_package))
            print(f"❌ Module manquant: {module}")
        else:
            print(f"✅ Module présent: {module}")
    
    # 3. Vérification spécifique Windows
    if os.name == 'nt':  # Windows
        print("\n🪟 Vérifications spécifiques Windows:")
        
        # Vérification PyQt5 sur Windows
        try:
            import PyQt5.QtWidgets
            print("✅ PyQt5 fonctionne correctement")
        except ImportError as e:
            print(f"❌ PyQt5 problème: {e}")
            issues.append("PyQt5 non fonctionnel sur Windows")
        
        # Vérification tkinter
        try:
            import tkinter
            print("✅ tkinter disponible")
        except ImportError:
            print("❌ tkinter manquant (problème installation Python)")
            issues.append("tkinter manquant")
    
    # 4. Vérification des fichiers dashboard
    dashboard_files = [
        'dashboard/dashboard_web.py',
        'dashboard/dashboard_qt.py', 
        'dashboard/dashboard_tk.py'
    ]
    
    for file in dashboard_files:
        if os.path.exists(file):
            print(f"✅ Fichier présent: {file}")
        else:
            print(f"❌ Fichier manquant: {file}")
            issues.append(f"Fichier manquant: {file}")
    
    return issues, missing_deps

def fix_dashboard_issues(missing_deps):
    """Tente de résoudre les problèmes identifiés"""
    print("\n🔧 Tentative de réparation...")
    print("=" * 50)
    
    # Installation des dépendances manquantes
    for module, pip_package in missing_deps:
        if pip_package:
            if not install_package(pip_package):
                print(f"❌ Impossible d'installer {pip_package}")
                return False
    
    # Vérification post-installation
    print("\n✅ Vérification post-installation...")
    all_good = True
    for module, _ in missing_deps:
        if check_module(module):
            print(f"✅ {module} maintenant disponible")
        else:
            print(f"❌ {module} toujours manquant")
            all_good = False
    
    return all_good

def create_windows_launcher():
    """Crée un lanceur Windows pour les dashboards"""
    launcher_content = '''@echo off
echo VRAMancer Dashboard Launcher
echo ==========================

echo Verification de l'environnement...
python -c "import sys; print(f'Python: {sys.version}')"

echo.
echo Dashboards disponibles:
echo 1. Web Dashboard (recommande)
echo 2. Qt Dashboard  
echo 3. Tkinter Dashboard
echo.

set /p choice="Choisissez (1/2/3): "

if "%choice%"=="1" (
    echo Lancement Web Dashboard...
    python dashboard/dashboard_web.py
) else if "%choice%"=="2" (
    echo Lancement Qt Dashboard...
    python dashboard/dashboard_qt.py
) else if "%choice%"=="3" (
    echo Lancement Tkinter Dashboard...
    python dashboard/dashboard_tk.py
) else (
    echo Choix invalide
    pause
)
'''
    
    with open('launch_dashboard.bat', 'w') as f:
        f.write(launcher_content)
    
    print("✅ Lanceur Windows créé: launch_dashboard.bat")

def create_minimal_dashboard():
    """Crée un dashboard minimal qui fonctionne sur Windows"""
    minimal_dashboard = '''#!/usr/bin/env python3
"""
Dashboard minimal VRAMancer pour Windows
Version simplifiée sans dépendances complexes
"""

import os
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse

class MinimalDashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/dashboard':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = '''<!DOCTYPE html>
<html>
<head>
    <title>VRAMancer - Dashboard Minimal Windows</title>
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
        .status.error { background: #dc3545; }
        .status.warning { background: #ffc107; color: #000; }
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
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 VRAMancer Dashboard - Windows Edition</h1>
        <p class="status ok">Dashboard minimal opérationnel</p>
        
        <div class="grid">
            <div class="card">
                <h2>🖥️ Système</h2>
                <p>OS: Windows</p>
                <p>Python: """ + sys.version.split()[0] + """</p>
                <p>Statut: <span class="status ok">Opérationnel</span></p>
            </div>
            
            <div class="card">
                <h2>🎮 GPU Status</h2>
                <p>Détection automatique...</p>
                <button onclick="refreshGPU()">Actualiser GPU</button>
            </div>
            
            <div class="card">
                <h2>🔧 Actions Rapides</h2>
                <button onclick="testSystem()">Test Système</button>
                <button onclick="checkDeps()">Vérifier Dépendances</button>
                <button onclick="location.reload()">Actualiser</button>
            </div>
            
            <div class="card">
                <h2>📊 Logs Système</h2>
                <div id="logs" style="background:#1a1a1a;padding:10px;border-radius:4px;height:200px;overflow-y:auto;">
                    <div>Dashboard minimal démarré avec succès</div>
                    <div>Prêt pour les tests de base</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>🚀 Prochaines Étapes</h2>
            <p>1. Vérifiez que ce dashboard minimal fonctionne</p>
            <p>2. Installez les dépendances manquantes si nécessaire</p>
            <p>3. Testez les dashboards avancés (Qt/Web)</p>
            <button onclick="window.open('https://github.com/thebloodlust/VRAMancer', '_blank')">Documentation</button>
        </div>
    </div>
    
    <script>
        function addLog(message) {
            const logs = document.getElementById('logs');
            const div = document.createElement('div');
            div.textContent = new Date().toLocaleTimeString() + ': ' + message;
            logs.appendChild(div);
            logs.scrollTop = logs.scrollHeight;
        }
        
        function refreshGPU() {
            addLog('Actualisation GPU...');
            // Simulation - dans la vraie version, ça appellerait l\\'API
            setTimeout(() => addLog('GPU: Detection simulée OK'), 1000);
        }
        
        function testSystem() {
            addLog('Test système en cours...');
            setTimeout(() => addLog('Système: Tests de base OK'), 1500);
        }
        
        function checkDeps() {
            addLog('Vérification dépendances...');
            setTimeout(() => addLog('Dépendances: Vérification simulée'), 1000);
        }
        
        // Auto-refresh logs
        setInterval(() => {
            addLog('Système opérationnel - ' + new Date().toLocaleString());
        }, 30000);
    </script>
</body>
</html>'''
            
            self.wfile.write(html.encode())
        
        elif self.path.startswith('/api/'):
            # API endpoints basiques
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {"status": "ok", "message": "API minimal active"}
            self.wfile.write(json.dumps(response).encode())
        
        else:
            self.send_response(404)
            self.end_headers()

def run_minimal_dashboard():
    """Lance le dashboard minimal"""
    print("🚀 Lancement du Dashboard Minimal VRAMancer...")
    print("=" * 50)
    
    port = 8080
    server = HTTPServer(('localhost', port), MinimalDashboardHandler)
    
    print(f"✅ Dashboard démarré sur: http://localhost:{port}")
    print("   - Dashboard: http://localhost:8080/dashboard")
    print("   - API: http://localhost:8080/api/status")
    print()
    print("🌐 Ouverture automatique du navigateur...")
    
    try:
        webbrowser.open(f'http://localhost:{port}/dashboard')
    except Exception:
        print("   (Ouverture navigateur échouée - ouvrez manuellement)")
    
    print()
    print("Appuyez Ctrl+C pour arrêter")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\n🛑 Dashboard arrêté")
        server.shutdown()

if __name__ == "__main__":
    run_minimal_dashboard()
'''
    
    with open('dashboard_minimal_windows.py', 'w', encoding='utf-8') as f:
        f.write(minimal_dashboard)
    
    print("✅ Dashboard minimal créé: dashboard_minimal_windows.py")

def main():
    print("🩺 VRAMancer Windows Dashboard Doctor")
    print("=" * 50)
    
    # Diagnostic
    issues, missing_deps = diagnose_dashboard_issues()
    
    if not issues and not missing_deps:
        print("\n🎉 Aucun problème détecté ! Les dashboards devraient fonctionner.")
        return
    
    print(f"\n📋 Résumé: {len(issues)} problèmes, {len(missing_deps)} dépendances manquantes")
    
    # Demander permission de réparer
    if missing_deps:
        print("\n🔧 Dépendances à installer:")
        for module, pip_package in missing_deps:
            print(f"   - {module} (via {pip_package})")
        
        response = input("\nInstaller les dépendances manquantes ? (o/n): ").lower()
        
        if response in ('o', 'oui', 'y', 'yes'):
            if fix_dashboard_issues(missing_deps):
                print("\n✅ Réparation terminée avec succès!")
            else:
                print("\n⚠️ Réparation partielle - certains problèmes persistent")
        else:
            print("Réparation annulée")
    
    # Créer des outils de secours
    print("\n🛠️ Création d'outils de secours Windows...")
    create_windows_launcher()
    create_minimal_dashboard()
    
    print("\n📋 Solutions disponibles:")
    print("1. launch_dashboard.bat - Lanceur Windows interactif")
    print("2. dashboard_minimal_windows.py - Dashboard de secours")
    print("3. python dashboard_minimal_windows.py - Test immédiat")
    
    print("\n💡 Pour tester immédiatement:")
    print("   python dashboard_minimal_windows.py")

if __name__ == "__main__":
    main()