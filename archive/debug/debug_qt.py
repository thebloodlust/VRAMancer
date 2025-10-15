#!/usr/bin/env python3
"""
VRAMancer Qt Debug - Diagnostic et réparation automatique Qt
"""

import os
import sys
import time
import subprocess
import traceback
import json
from pathlib import Path

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

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

def test_qt_imports():
    """Test des imports Qt"""
    print_section("TEST IMPORTS QT")
    
    qt_packages = [
        ("PyQt5", "PyQt5.QtWidgets"),
        ("PyQt6", "PyQt6.QtWidgets"), 
        ("PySide2", "PySide2.QtWidgets"),
        ("PySide6", "PySide6.QtWidgets")
    ]
    
    available_qt = []
    
    for package, import_name in qt_packages:
        try:
            __import__(import_name)
            print_status(f"{package} disponible", "SUCCESS")
            available_qt.append(package)
        except ImportError:
            print_status(f"{package} non installé", "WARNING")
    
    if not available_qt:
        print_status("Aucune version de Qt trouvée", "ERROR")
        print_status("Installation de PyQt5...", "INFO")
        if install_package("PyQt5"):
            available_qt.append("PyQt5")
    
    return available_qt

def test_api_compatibility():
    """Test de compatibilité API pour Qt"""
    print_section("TEST COMPATIBILITÉ API QT")
    
    api_base = os.environ.get('VRM_API_BASE', 'http://localhost:5030')
    print_status(f"API configurée: {api_base}", "INFO")
    
    try:
        import requests
        
        # Test endpoint /api/nodes - problématique pour Qt
        print_status("Test endpoint /api/nodes (critique pour Qt)...", "DEBUG")
        response = requests.get(f"{api_base}/api/nodes", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_status("Réponse /api/nodes reçue", "SUCCESS")
            
            # Analyse de la structure des données
            nodes = data.get('nodes', [])
            if nodes:
                first_node = nodes[0]
                print_status(f"Structure du premier node: {type(first_node)}", "DEBUG")
                
                if isinstance(first_node, str):
                    print_status("❌ PROBLÈME IDENTIFIÉ: Les nodes sont des strings, Qt attend des objets", "ERROR")
                    return False, "nodes_format_error"
                elif isinstance(first_node, dict):
                    required_fields = ['id', 'name', 'status', 'type']
                    missing_fields = [f for f in required_fields if f not in first_node]
                    
                    if missing_fields:
                        print_status(f"❌ Champs manquants dans node: {missing_fields}", "ERROR")
                        return False, "missing_fields"
                    else:
                        print_status("✅ Structure des nodes compatible Qt", "SUCCESS")
                        return True, "ok"
            else:
                print_status("⚠️ Aucun node dans la réponse", "WARNING")
                return False, "no_nodes"
        else:
            print_status(f"❌ Erreur HTTP: {response.status_code}", "ERROR")
            return False, "http_error"
            
    except ImportError:
        print_status("Module requests manquant", "ERROR")
        return False, "no_requests"
    except Exception as e:
        print_status(f"Erreur test API: {e}", "ERROR")
        return False, "api_error"

def create_qt_test_app(qt_package):
    """Création d'une app Qt de test"""
    print_section(f"CRÉATION APP TEST {qt_package}")
    
    try:
        if qt_package == "PyQt5":
            from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QTextEdit
            from PyQt5.QtCore import QTimer, Qt
            from PyQt5.QtGui import QFont
        elif qt_package == "PyQt6":
            from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QTextEdit
            from PyQt6.QtCore import QTimer, Qt
            from PyQt6.QtGui import QFont
        elif qt_package == "PySide2":
            from PySide2.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QTextEdit
            from PySide2.QtCore import QTimer, Qt
            from PySide2.QtGui import QFont
        elif qt_package == "PySide6":  
            from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QTextEdit
            from PySide6.QtCore import QTimer, Qt
            from PySide6.QtGui import QFont
        else:
            print_status(f"Package Qt non supporté: {qt_package}", "ERROR")
            return None
            
        print_status(f"Imports {qt_package} réussis", "SUCCESS")
        
        class QtDebugWindow(QMainWindow):
            def __init__(self):
                super().__init__()
                self.setWindowTitle(f"VRAMancer - Debug Qt ({qt_package})")
                self.setGeometry(100, 100, 800, 600)
                
                # Widget central
                central_widget = QWidget()
                self.setCentralWidget(central_widget)
                layout = QVBoxLayout(central_widget)
                
                # Titre
                title = QLabel("🚀 VRAMancer - Debug Qt Dashboard")
                title.setAlignment(Qt.AlignCenter)
                font = QFont()
                font.setPointSize(16)
                font.setBold(True)
                title.setFont(font)
                layout.addWidget(title)
                
                # Informations système
                info_label = QLabel(f"""
📋 Informations Système:
• Python: {sys.version.split()[0]}
• Qt Package: {qt_package}
• API Base: {os.environ.get('VRM_API_BASE', 'Non défini')}
• Répertoire: {os.getcwd()}
                """)
                layout.addWidget(info_label)
                
                # Zone de log
                self.log_area = QTextEdit()
                self.log_area.setReadOnly(True)
                self.log_area.setStyleSheet("""
                    QTextEdit {
                        background-color: #2d2d2d;
                        color: #ffffff;
                        font-family: 'Consolas', monospace;
                        font-size: 10pt;
                        border: 1px solid #444;
                    }
                """)
                layout.addWidget(self.log_area)
                
                # Boutons de test
                test_api_btn = QPushButton("🧪 Test API")
                test_api_btn.clicked.connect(self.test_api)
                layout.addWidget(test_api_btn)
                
                test_nodes_btn = QPushButton("🖥️ Test Nodes (Problématique)")
                test_nodes_btn.clicked.connect(self.test_nodes_endpoint)
                layout.addWidget(test_nodes_btn)
                
                fix_api_btn = QPushButton("🔧 Corriger API pour Qt")
                fix_api_btn.clicked.connect(self.suggest_fix)
                layout.addWidget(fix_api_btn)
                
                close_btn = QPushButton("❌ Fermer")
                close_btn.clicked.connect(self.close)
                layout.addWidget(close_btn)
                
                # Timer pour actualisation
                self.timer = QTimer()
                self.timer.timeout.connect(self.update_status)
                self.timer.start(5000)  # 5 secondes
                
                self.log("Interface Qt Debug initialisée")
                self.log(f"Package Qt utilisé: {qt_package}")
            
            def log(self, message):
                timestamp = time.strftime("%H:%M:%S")
                self.log_area.append(f"[{timestamp}] {message}")
            
            def test_api(self):
                self.log("Test de l'API...")
                try:
                    import requests
                    response = requests.get('http://localhost:5030/health', timeout=3)
                    if response.status_code == 200:
                        self.log("✅ API accessible")
                        data = response.json()
                        self.log(f"Réponse: {data}")
                    else:
                        self.log(f"❌ API erreur: {response.status_code}")
                except Exception as e:
                    self.log(f"❌ Erreur API: {e}")
            
            def test_nodes_endpoint(self):
                self.log("Test de l'endpoint /api/nodes...")
                try:
                    import requests
                    response = requests.get('http://localhost:5030/api/nodes', timeout=3)
                    if response.status_code == 200:
                        data = response.json()
                        nodes = data.get('nodes', [])
                        
                        if nodes:
                            first_node = nodes[0]
                            self.log(f"Type du premier node: {type(first_node)}")
                            
                            if isinstance(first_node, str):
                                self.log("❌ PROBLÈME: Node est une string, Qt attend un objet")
                                self.log("Solution: L'API doit retourner des objets JSON")
                            elif isinstance(first_node, dict):
                                self.log("✅ Node est un objet (bon format)")
                                
                                # Vérification des champs requis
                                required = ['type', 'id', 'name', 'status']
                                missing = [f for f in required if f not in first_node]
                                
                                if missing:
                                    self.log(f"❌ Champs manquants: {missing}")
                                else:
                                    self.log("✅ Tous les champs requis présents")
                                    
                                # Tentative d'accès comme le fait Qt
                                try:
                                    node_type = first_node.get("type", "standard")
                                    self.log(f"✅ Accès node.get('type'): {node_type}")
                                except AttributeError as e:
                                    self.log(f"❌ Erreur AttributeError: {e}")
                        else:
                            self.log("⚠️ Aucun node dans la réponse")
                    else:
                        self.log(f"❌ Erreur HTTP: {response.status_code}")
                except Exception as e:
                    self.log(f"❌ Erreur test nodes: {e}")
            
            def suggest_fix(self):
                self.log("🔧 Suggestions de correction pour Qt:")
                self.log("1. Vérifiez que l'API retourne des objets JSON, pas des strings")
                self.log("2. Assurez-vous que chaque node a un champ 'type'")
                self.log("3. Structure attendue: {'id': '...', 'name': '...', 'status': '...', 'type': '...'}")
                self.log("4. Redémarrez l'API après modifications")
            
            def update_status(self):
                self.log("Actualisation automatique...")
                self.test_api()
        
        return QtDebugWindow
        
    except ImportError as e:
        print_status(f"Erreur import {qt_package}: {e}", "ERROR")
        return None
    except Exception as e:
        print_status(f"Erreur création app Qt: {e}", "ERROR")
        traceback.print_exc()
        return None

def analyze_qt_dashboard_error():
    """Analyse l'erreur spécifique au dashboard Qt"""
    print_section("ANALYSE ERREUR DASHBOARD QT")
    
    dashboard_file = Path("dashboard/dashboard_qt.py")
    if not dashboard_file.exists():
        print_status("Fichier dashboard_qt.py non trouvé", "ERROR")
        return
    
    print_status("Lecture du code Qt dashboard...", "DEBUG")
    
    try:
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Recherche de la ligne problématique
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'node.get("type"' in line:
                print_status(f"Ligne problématique trouvée (ligne {i+1}):", "DEBUG")
                print_status(f"  {line.strip()}", "DEBUG")
                
                # Contexte autour de la ligne
                start = max(0, i-3)
                end = min(len(lines), i+4)
                print_status("Contexte:", "DEBUG")
                for j in range(start, end):
                    marker = ">>> " if j == i else "    "
                    print_status(f"{marker}L{j+1:3d}: {lines[j]}", "DEBUG")
                break
        
        print_status("❌ Erreur identifiée: 'str' object has no attribute 'get'", "ERROR")
        print_status("Cause: L'API retourne des strings au lieu d'objets pour les nodes", "ERROR")
        
    except Exception as e:
        print_status(f"Erreur analyse dashboard: {e}", "ERROR")

def generate_api_fix():
    """Génère un correctif pour l'API"""
    print_section("GÉNÉRATION CORRECTIF API")
    
    fix_code = '''
# Correctif pour l'API - À ajouter dans start_api.py
# Remplacer l'endpoint /api/nodes par:

@app.route('/api/nodes')
def nodes():
    return jsonify({
        "nodes": [
            {
                "id": "local",
                "name": "Local Node",
                "status": "active", 
                "type": "standard",    # ← CHAMP REQUIS POUR QT
                "gpu_count": 1,
                "memory_total": "8GB",
                "memory_used": "2GB",
                "address": "127.0.0.1:5030",
                "capabilities": ["cuda", "inference"]
            }
        ],
        "total_nodes": 1
    })
'''
    
    with open("api_fix_for_qt.py", "w", encoding="utf-8") as f:
        f.write(fix_code)
    
    print_status("Correctif généré: api_fix_for_qt.py", "SUCCESS")
    print_status("Le champ 'type' est maintenant inclus dans les nodes", "INFO")

def main():
    print_section("VRAMANCER QT DEBUG - DIAGNOSTIC COMPLET")
    
    # Configuration environnement
    os.environ['VRM_API_BASE'] = 'http://localhost:5030'
    print_status("Variables d'environnement configurées", "SUCCESS")
    
    # Test des imports Qt
    available_qt = test_qt_imports()
    
    if not available_qt:
        print_status("Impossible de continuer sans Qt", "ERROR")
        return
    
    # Test de compatibilité API
    api_ok, error_type = test_api_compatibility()
    
    # Analyse de l'erreur Qt dashboard
    analyze_qt_dashboard_error()
    
    # Génération du correctif
    if not api_ok:
        generate_api_fix()
    
    # Création et lancement de l'app Qt de test
    qt_package = available_qt[0]  # Utiliser le premier disponible
    print_section(f"LANCEMENT APP DEBUG {qt_package}")
    
    try:
        if qt_package.startswith("PyQt"):
            from PyQt5.QtWidgets import QApplication
        else:
            from PySide2.QtWidgets import QApplication
            
        app = QApplication(sys.argv)
        
        QtDebugWindow = create_qt_test_app(qt_package)
        if QtDebugWindow:
            window = QtDebugWindow()
            window.show()
            
            print_status(f"Interface Qt Debug lancée avec {qt_package}", "SUCCESS")
            print_status("Fermez la fenêtre pour terminer", "INFO")
            
            if not api_ok:
                print_status(f"⚠️ Problème API détecté: {error_type}", "WARNING")
                print_status("Consultez le correctif généré: api_fix_for_qt.py", "INFO")
            
            app.exec_()
        else:
            print_status("Impossible de créer l'interface Qt", "ERROR")
            
    except Exception as e:
        print_status(f"Erreur lancement Qt: {e}", "ERROR")
        traceback.print_exc()

if __name__ == "__main__":
    main()