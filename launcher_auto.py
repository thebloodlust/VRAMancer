#!/usr/bin/env python3
"""Launcher Auto VRAMancer - Detection automatique des interfaces."""

import sys
import os
import subprocess
import importlib
import time
import requests
from datetime import datetime

def test_import(module_name):
    """Test si un module peut etre importe."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def test_api():
    """Test si l'API VRAMancer est accessible."""
    try:
        response = requests.get("http://localhost:5030/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def launch_auto():
    """Lance automatiquement la meilleure interface disponible."""
    print("=" * 80)
    print("                    VRAMANCER - LAUNCHER AUTO")
    print("=" * 80)
    print()
    
    print("Detection des interfaces disponibles...")
    print()
    
    # Test API d'abord
    api_available = test_api()
    print(f"API VRAMancer (port 5030): {'✓ Disponible' if api_available else '❌ Non accessible'}")
    
    if not api_available:
        print()
        print("⚠️  L'API VRAMancer n'est pas accessible!")
        print("   Lancez d'abord l'API avec l'option 1 du menu principal")
        print("   ou executez: api_permanente.bat")
        print()
        input("Appuyez sur Entree pour continuer...")
        return
    
    # Detection des interfaces
    interfaces = []
    
    # Test Qt
    if test_import("PyQt5.QtWidgets") or test_import("PyQt6.QtWidgets") or test_import("PySide2.QtWidgets") or test_import("PySide6.QtWidgets"):
        interfaces.append(("Qt Dashboard", "dashboard\\dashboard_qt.py", "Interface native Qt avec monitoring"))
        print("Qt (PyQt/PySide): ✓ Disponible")
    else:
        print("Qt (PyQt/PySide): ❌ Non disponible")
    
    # Test Tkinter
    if test_import("tkinter"):
        interfaces.append(("Tkinter Dashboard", "dashboard\\dashboard_tk.py", "Interface Tkinter integree"))
        print("Tkinter: ✓ Disponible")
    else:
        print("Tkinter: ❌ Non disponible")
    
    # Test Flask (pour interfaces web)
    if test_import("flask"):
        interfaces.append(("Dashboard Web", "dashboard\\dashboard_web.py", "Interface web sur navigateur"))
        interfaces.append(("Debug Web Ultra", "debug_web_ultra.py", "Interface de debug avancee"))
        print("Flask (interfaces web): ✓ Disponible")
    else:
        print("Flask (interfaces web): ❌ Non disponible")
    
    # Toujours disponible - CLI
    interfaces.append(("CLI Dashboard", "dashboard\\dashboard_cli.py", "Interface ligne de commande"))
    print("CLI Dashboard: ✓ Toujours disponible")
    
    print()
    
    if not interfaces:
        print("❌ Aucune interface graphique disponible!")
        print("   Installez les dependances necessaires:")
        print("   pip install PyQt5 flask requests")
        return
    
    print(f"Interfaces detectees: {len(interfaces)}")
    print()
    
    # Selection automatique de la meilleure interface
    if len(interfaces) == 1:
        selected = interfaces[0]
        print(f"Lancement automatique de: {selected[0]}")
    else:
        print("Selection de l'interface:")
        for i, (name, path, desc) in enumerate(interfaces, 1):
            print(f"  {i}. {name} - {desc}")
        print()
        
        try:
            choice = input("Choisissez une interface (1-{}) ou Entree pour auto: ".format(len(interfaces)))
            if choice.strip() == "":
                # Auto: privilegie Qt > Tkinter > Web > CLI
                priority = ["Qt Dashboard", "Tkinter Dashboard", "Dashboard Web", "CLI Dashboard"]
                selected = interfaces[0]  # par defaut
                for pref in priority:
                    for interface in interfaces:
                        if interface[0] == pref:
                            selected = interface
                            break
                    else:
                        continue
                    break
                print(f"Selection automatique: {selected[0]}")
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(interfaces):
                    selected = interfaces[idx]
                else:
                    print("Choix invalide, utilisation de l'interface par defaut")
                    selected = interfaces[0]
        except (ValueError, KeyboardInterrupt):
            print("Selection par defaut")
            selected = interfaces[0]
    
    print()
    print(f"Lancement de {selected[0]}...")
    print(f"Description: {selected[2]}")
    print()
    
    try:
        # Lancement de l'interface selectionnee
        if os.path.exists(selected[1]):
            subprocess.run([sys.executable, selected[1]], check=True)
        else:
            print(f"❌ Fichier non trouve: {selected[1]}")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du lancement: {e}")
    except KeyboardInterrupt:
        print("Interruption utilisateur")
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")

if __name__ == "__main__":
    launch_auto()