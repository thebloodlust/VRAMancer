#!/usr/bin/env python3
"""
System Tray Console Mode - Pour environnements sans GUI
Alternative au system tray graphique
"""

import subprocess
import sys
import time
import os

class VRAMancerConsoleHub:
    def __init__(self):
        self.api_process = None
        self.running = True
        
    def show_logo(self):
        print("=" * 50)
        print("    🚀 VRAMANCER CONSOLE HUB 🚀")
        print("=" * 50)
        print("Hub central pour environnements sans GUI")
        print("")
        
    def check_api(self):
        """Vérifie si l'API est en cours d'exécution"""
        try:
            import requests
            response = requests.get('http://localhost:5030/health', timeout=2)
            return response.status_code == 200
        except:
            return False
            
    def start_api(self):
        """Lance l'API en arrière-plan"""
        if not self.check_api():
            print("🔌 Lancement API VRAMancer...")
            env_python = ".venv/bin/python" if os.path.exists(".venv/bin/python") else "python"
            self.api_process = subprocess.Popen([env_python, "api_simple.py"])
            time.sleep(3)
            
            if self.check_api():
                print("✅ API opérationnelle sur http://localhost:5030")
            else:
                print("⚠️ API en cours de démarrage...")
        else:
            print("✅ API déjà active")
    
    def show_menu(self):
        print("\n📋 INTERFACES DISPONIBLES:")
        print("[1] Dashboard Web Avancé (Supervision)")
        print("[2] Dashboard Mobile (Responsive)")  
        print("[3] Lancer tous les dashboards web")
        print("[4] Status API et endpoints")
        print("[5] Test GPU RTX 4060")
        print("[0] Quitter")
        print("")
        
    def launch_web_advanced(self):
        """Lance le dashboard web avancé"""
        print("🌐 Lancement Dashboard Web Avancé...")
        print("📍 URL: http://localhost:5000")
        env_python = ".venv/bin/python" if os.path.exists(".venv/bin/python") else "python"
        subprocess.Popen([env_python, "dashboard/dashboard_web_advanced.py"])
        
    def launch_mobile(self):
        """Lance le dashboard mobile"""
        print("📱 Lancement Dashboard Mobile...")
        print("📍 URL: http://localhost:5003")
        env_python = ".venv/bin/python" if os.path.exists(".venv/bin/python") else "python"
        subprocess.Popen([env_python, "mobile/dashboard_mobile.py"])
        
    def launch_all_web(self):
        """Lance tous les dashboards web"""
        print("🚀 Lancement tous les dashboards web...")
        self.launch_web_advanced()
        time.sleep(2)
        self.launch_mobile()
        print("✅ Dashboards lancés:")
        print("   🌐 Web Advanced: http://localhost:5000")
        print("   📱 Mobile: http://localhost:5003")
        
    def show_api_status(self):
        """Affiche le status de l'API et des endpoints"""
        print("🔍 Test des endpoints API...")
        try:
            import requests
            
            endpoints = [
                ("Health", "/health"),
                ("System", "/api/system"), 
                ("GPU", "/api/gpu"),
                ("Nodes", "/api/nodes")
            ]
            
            for name, endpoint in endpoints:
                try:
                    response = requests.get(f'http://localhost:5030{endpoint}', timeout=3)
                    status = "✅ OK" if response.status_code == 200 else f"❌ {response.status_code}"
                    print(f"   {name}: {status}")
                except:
                    print(f"   {name}: ❌ Erreur")
                    
        except ImportError:
            print("❌ Module requests non disponible")
            
    def test_gpu(self):
        """Test détection GPU RTX 4060"""
        print("🎮 Test détection GPU RTX 4060...")
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
                print(f"✅ CUDA disponible: {gpu_count} GPU(s)")
                print(f"🎮 GPU principal: {gpu_name}")
                
                # Test mémoire
                props = torch.cuda.get_device_properties(0)
                total_memory = props.total_memory / (1024**3)
                print(f"💾 VRAM totale: {total_memory:.1f} GB")
                
                if torch.cuda.is_initialized():
                    used_memory = torch.cuda.memory_allocated(0) / (1024**3)
                    print(f"📊 VRAM utilisée: {used_memory:.3f} GB")
                    
            else:
                print("❌ CUDA non disponible")
        except ImportError:
            print("❌ PyTorch non installé")
            
    def run(self):
        """Boucle principale du hub"""
        self.show_logo()
        self.start_api()
        
        while self.running:
            self.show_menu()
            
            try:
                choice = input("👉 Votre choix: ").strip()
                
                if choice == "1":
                    self.launch_web_advanced()
                elif choice == "2":
                    self.launch_mobile()
                elif choice == "3":
                    self.launch_all_web()
                elif choice == "4":
                    self.show_api_status()  
                elif choice == "5":
                    self.test_gpu()
                elif choice == "0":
                    print("👋 Arrêt VRAMancer Console Hub")
                    self.running = False
                else:
                    print("❌ Choix invalide")
                    
                if choice != "0":
                    input("\n⏸️ Appuyez sur Entrée pour continuer...")
                    
            except KeyboardInterrupt:
                print("\n👋 Arrêt via Ctrl+C")
                self.running = False
                
        # Nettoyage
        if self.api_process:
            try:
                self.api_process.terminate()
                print("🔌 API arrêtée")
            except:
                pass

if __name__ == "__main__":
    hub = VRAMancerConsoleHub()
    hub.run()