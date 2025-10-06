#!/usr/bin/env python3
"""
Lanceur intelligent VRAMancer pour Windows
Détecte automatiquement les problèmes et propose des solutions
"""

import os
import sys
import subprocess
import importlib.util
import webbrowser
import platform
from pathlib import Path

class VRAMancerLauncher:
    def __init__(self):
        self.is_windows = platform.system() == "Windows"
        self.python_exe = sys.executable
        self.missing_deps = []
        self.dashboard_status = {}
        
    def check_python_version(self):
        """Vérifie la version de Python"""
        version = sys.version_info
        print(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
        
        if version < (3, 8):
            print("⚠️  ATTENTION: Python 3.8+ recommandé")
            return False
        return True
    
    def check_dependencies(self):
        """Vérifie toutes les dépendances critiques"""
        print("🔍 Vérification des dépendances...")
        
        # Dépendances pour chaque dashboard
        deps = {
            'web': ['flask', 'flask_socketio', 'requests'],
            'qt': ['PyQt5', 'PyQt5.QtWidgets'],
            'tk': ['tkinter'],
            'core': ['numpy', 'psutil', 'torch']
        }
        
        for dashboard, modules in deps.items():
            status = True
            missing = []
            
            for module in modules:
                try:
                    if '.' in module:  # Sous-module
                        parent, child = module.split('.', 1)
                        parent_spec = importlib.util.find_spec(parent)
                        if parent_spec:
                            # Tenter d'importer le sous-module
                            exec(f"import {module}")
                    else:
                        importlib.import_module(module)
                    print(f"  ✅ {module}")
                except ImportError:
                    print(f"  ❌ {module}")
                    missing.append(module)
                    status = False
            
            self.dashboard_status[dashboard] = {
                'available': status,
                'missing': missing
            }
        
        return any(d['available'] for d in self.dashboard_status.values())
    
    def install_missing_dependencies(self):
        """Installe les dépendances manquantes"""
        print("🔧 Installation des dépendances manquantes...")
        
        # Map des modules vers packages pip
        pip_packages = {
            'flask': 'flask',
            'flask_socketio': 'flask-socketio',
            'PyQt5': 'PyQt5',
            'requests': 'requests',
            'numpy': 'numpy',
            'psutil': 'psutil',
            'torch': 'torch'
        }
        
        to_install = set()
        for dashboard, info in self.dashboard_status.items():
            if not info['available']:
                for missing in info['missing']:
                    if missing in pip_packages:
                        to_install.add(pip_packages[missing])
        
        if to_install:
            print(f"📦 Installation de: {', '.join(to_install)}")
            try:
                cmd = [self.python_exe, "-m", "pip", "install"] + list(to_install)
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print("✅ Installation réussie!")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Erreur installation: {e}")
                print(f"Sortie: {e.stdout}")
                print(f"Erreur: {e.stderr}")
                return False
        return True
    
    def launch_dashboard(self, dashboard_type):
        """Lance un dashboard spécifique"""
        dashboard_files = {
            'web': 'dashboard/dashboard_web.py',
            'qt': 'dashboard/dashboard_qt.py',
            'tk': 'dashboard/dashboard_tk.py',
            'minimal': 'dashboard_minimal_windows.py'
        }
        
        if dashboard_type not in dashboard_files:
            print(f"❌ Dashboard type inconnu: {dashboard_type}")
            return False
        
        dashboard_file = dashboard_files[dashboard_type]
        
        if not os.path.exists(dashboard_file):
            print(f"❌ Fichier manquant: {dashboard_file}")
            return False
        
        print(f"🚀 Lancement du dashboard {dashboard_type}...")
        
        try:
            if dashboard_type == 'web':
                # Pour le web dashboard, ouvrir aussi le navigateur
                import threading
                def open_browser():
                    import time
                    time.sleep(2)  # Attendre que le serveur démarre
                    webbrowser.open('http://localhost:5000')
                
                threading.Thread(target=open_browser, daemon=True).start()
            
            # Lancer le dashboard
            subprocess.run([self.python_exe, dashboard_file], check=True)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur lancement: {e}")
            return False
        except KeyboardInterrupt:
            print("\n🛑 Dashboard arrêté par l'utilisateur")
            return True
    
    def show_menu(self):
        """Affiche le menu principal"""
        print("\n" + "="*50)
        print("🧠 VRAMancer Dashboard Launcher")
        print("="*50)
        
        # Afficher le statut des dashboards
        print("\n📊 Dashboards disponibles:")
        
        icons = {'web': '🌐', 'qt': '🖥️', 'tk': '🪟'}
        
        for dashboard, info in self.dashboard_status.items():
            if dashboard in icons:
                icon = icons[dashboard]
                status = "✅ Prêt" if info['available'] else f"❌ Manque: {', '.join(info['missing'])}"
                print(f"  {icon} {dashboard.upper()}: {status}")
        
        print("\n🛠️ Actions disponibles:")
        print("  1. Web Dashboard (recommandé)")
        print("  2. Qt Dashboard")
        print("  3. Tkinter Dashboard")
        print("  4. Dashboard Minimal (secours)")
        print("  5. Installer dépendances manquantes")
        print("  6. Diagnostic complet")
        print("  0. Quitter")
        
        return input("\nChoisissez une option (0-6): ").strip()
    
    def run_diagnostic(self):
        """Lance le diagnostic complet"""
        print("\n🩺 Diagnostic complet VRAMancer...")
        
        # Exécuter le script de diagnostic
        if os.path.exists('fix_windows_dashboard.py'):
            try:
                subprocess.run([self.python_exe, 'fix_windows_dashboard.py'], check=True)
            except subprocess.CalledProcessError:
                print("❌ Erreur lors du diagnostic")
        else:
            print("❌ Script de diagnostic manquant")
    
    def main_loop(self):
        """Boucle principale du lanceur"""
        print("🎯 Initialisation VRAMancer Launcher...")
        
        # Vérifications initiales
        if not self.check_python_version():
            print("❌ Version Python incompatible")
            return
        
        self.check_dependencies()
        
        while True:
            choice = self.show_menu()
            
            if choice == '0':
                print("👋 Au revoir!")
                break
            
            elif choice == '1':
                if self.dashboard_status['web']['available']:
                    self.launch_dashboard('web')
                else:
                    print("❌ Web dashboard non disponible. Installez les dépendances d'abord.")
            
            elif choice == '2':
                if self.dashboard_status['qt']['available']:
                    self.launch_dashboard('qt')
                else:
                    print("❌ Qt dashboard non disponible. Installez PyQt5 d'abord.")
            
            elif choice == '3':
                if self.dashboard_status['tk']['available']:
                    self.launch_dashboard('tk')
                else:
                    print("❌ Tkinter dashboard non disponible.")
            
            elif choice == '4':
                self.launch_dashboard('minimal')
            
            elif choice == '5':
                if self.install_missing_dependencies():
                    print("✅ Redémarrez le lanceur pour voir les changements")
                    self.check_dependencies()  # Re-vérifier
            
            elif choice == '6':
                self.run_diagnostic()
            
            else:
                print("❌ Choix invalide")
            
            input("\nAppuyez sur Entrée pour continuer...")

def main():
    """Point d'entrée principal"""
    try:
        launcher = VRAMancerLauncher()
        launcher.main_loop()
    except KeyboardInterrupt:
        print("\n👋 Lanceur interrompu")
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()