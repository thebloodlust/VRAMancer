#!/usr/bin/env python3
"""
VRAMancer - Interface Tkinter Simple
Fonctionne indépendamment sans imports complexes
"""

import tkinter as tk
from tkinter import ttk, messagebox
import requests
import threading
import time
import os

class VRAMancerTkGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VRAMancer - Dashboard Tkinter")
        self.root.geometry("800x600")
        self.root.configure(bg="#2d2d2d")
        
        # Variables
        self.api_base = os.environ.get('VRM_API_BASE', 'http://localhost:5030')
        self.api_status = tk.StringVar(value="⚪ Vérification...")
        self.gpu_info = tk.StringVar(value="Chargement...")
        self.nodes_info = tk.StringVar(value="Chargement...")
        
        self.setup_ui()
        self.start_monitoring()
    
    def setup_ui(self):
        """Configuration de l'interface utilisateur"""
        
        # Titre
        title_frame = tk.Frame(self.root, bg="#2d2d2d")
        title_frame.pack(pady=10)
        
        tk.Label(title_frame, text="🚀 VRAMancer Dashboard", 
                font=("Arial", 16, "bold"), 
                fg="#00bfff", bg="#2d2d2d").pack()
        
        tk.Label(title_frame, text=f"API: {self.api_base}", 
                font=("Arial", 10), 
                fg="#888", bg="#2d2d2d").pack()
        
        # Status API
        status_frame = tk.Frame(self.root, bg="#2d2d2d")
        status_frame.pack(pady=10, fill="x", padx=20)
        
        tk.Label(status_frame, text="Status API:", 
                font=("Arial", 12, "bold"), 
                fg="white", bg="#2d2d2d").pack(side="left")
        
        tk.Label(status_frame, textvariable=self.api_status, 
                font=("Arial", 12), 
                fg="white", bg="#2d2d2d").pack(side="left", padx=10)
        
        # Boutons d'action
        button_frame = tk.Frame(self.root, bg="#2d2d2d")
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="🔄 Actualiser API", 
                command=self.check_api_manual,
                bg="#007acc", fg="white", 
                font=("Arial", 10), padx=20).pack(side="left", padx=5)
        
        tk.Button(button_frame, text="🌐 Ouvrir Web", 
                command=self.open_web_dashboard,
                bg="#28a745", fg="white", 
                font=("Arial", 10), padx=20).pack(side="left", padx=5)
        
        tk.Button(button_frame, text="🔧 Test API", 
                command=self.test_all_endpoints,
                bg="#ffc107", fg="black", 
                font=("Arial", 10), padx=20).pack(side="left", padx=5)
        
        # Informations détaillées
        info_frame = tk.Frame(self.root, bg="#2d2d2d")
        info_frame.pack(pady=20, fill="both", expand=True, padx=20)
        
        # GPU Info
        gpu_frame = tk.LabelFrame(info_frame, text="🎮 GPU Information", 
                                 fg="#00bfff", bg="#2d2d2d", 
                                 font=("Arial", 12, "bold"))
        gpu_frame.pack(fill="x", pady=5)
        
        tk.Label(gpu_frame, textvariable=self.gpu_info, 
                justify="left", fg="white", bg="#2d2d2d",
                font=("Arial", 10)).pack(anchor="w", padx=10, pady=5)
        
        # Nodes Info
        nodes_frame = tk.LabelFrame(info_frame, text="🖥️ Cluster Nodes", 
                                   fg="#00bfff", bg="#2d2d2d", 
                                   font=("Arial", 12, "bold"))
        nodes_frame.pack(fill="x", pady=5)
        
        tk.Label(nodes_frame, textvariable=self.nodes_info, 
                justify="left", fg="white", bg="#2d2d2d",
                font=("Arial", 10)).pack(anchor="w", padx=10, pady=5)
        
        # Log area
        log_frame = tk.LabelFrame(info_frame, text="📝 Log", 
                                 fg="#00bfff", bg="#2d2d2d", 
                                 font=("Arial", 12, "bold"))
        log_frame.pack(fill="both", expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, height=8, bg="#1a1a1a", 
                               fg="white", font=("Consolas", 9))
        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y", pady=5)
        
        self.log("Interface Tkinter initialisée")
        self.log(f"API configurée: {self.api_base}")
    
    def log(self, message):
        """Ajouter un message au log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def check_api_manual(self):
        """Vérification manuelle de l'API"""
        self.log("Vérification manuelle de l'API...")
        threading.Thread(target=self.check_api, daemon=True).start()
    
    def check_api(self):
        """Vérification de l'état de l'API"""
        try:
            response = requests.get(f"{self.api_base}/health", timeout=3)
            if response.status_code == 200:
                self.api_status.set("✅ API Active")
                self.log("API Health Check: OK")
                self.update_gpu_info()
                self.update_nodes_info()
                return True
            else:
                self.api_status.set(f"❌ Erreur {response.status_code}")
                self.log(f"API Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.api_status.set("❌ API Inaccessible")
            self.log(f"API Connection Error: {e}")
        return False
    
    def update_gpu_info(self):
        """Mise à jour des informations GPU"""
        try:
            response = requests.get(f"{self.api_base}/api/gpu/info", timeout=3)
            if response.status_code == 200:
                data = response.json()
                gpu_text = f"GPUs détectés: {len(data.get('gpus', []))}\n"
                for i, gpu in enumerate(data.get('gpus', [])):
                    gpu_text += f"GPU {i}: {gpu.get('name', 'Unknown')} - {gpu.get('memory', 'N/A')}\n"
                gpu_text += f"Mémoire totale: {data.get('total_memory', 'N/A')}"
                self.gpu_info.set(gpu_text)
                self.log("GPU Info mis à jour")
        except Exception as e:
            self.gpu_info.set(f"Erreur GPU: {e}")
            self.log(f"GPU Update Error: {e}")
    
    def update_nodes_info(self):
        """Mise à jour des informations de noeuds"""
        try:
            response = requests.get(f"{self.api_base}/api/nodes", timeout=3)
            if response.status_code == 200:
                data = response.json()
                nodes_text = f"Noeuds actifs: {data.get('total_nodes', 0)}\n"
                for node in data.get('nodes', []):
                    nodes_text += f"• {node.get('name', 'Unknown')}: {node.get('status', 'unknown')}\n"
                    nodes_text += f"  GPU: {node.get('gpu_count', 0)}, Mém: {node.get('memory_used', 'N/A')}/{node.get('memory_total', 'N/A')}\n"
                self.nodes_info.set(nodes_text)
                self.log("Nodes Info mis à jour")
        except Exception as e:
            self.nodes_info.set(f"Erreur Nodes: {e}")
            self.log(f"Nodes Update Error: {e}")
    
    def open_web_dashboard(self):
        """Ouvrir le dashboard web"""
        import webbrowser
        web_url = "http://localhost:8080"
        webbrowser.open(web_url)
        self.log(f"Ouverture dashboard web: {web_url}")
    
    def test_all_endpoints(self):
        """Test de tous les endpoints API"""
        self.log("Test de tous les endpoints...")
        endpoints = ["/health", "/api/status", "/api/gpu/info", "/api/nodes", "/api/telemetry.bin"]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.api_base}{endpoint}", timeout=2)
                self.log(f"✅ {endpoint}: {response.status_code}")
            except Exception as e:
                self.log(f"❌ {endpoint}: {e}")
    
    def start_monitoring(self):
        """Démarrage du monitoring en arrière-plan"""
        def monitor():
            while True:
                self.check_api()
                time.sleep(30)  # Vérification toutes les 30 secondes
        
        threading.Thread(target=monitor, daemon=True).start()
        self.log("Monitoring automatique démarré (30s)")
    
    def run(self):
        """Lancement de l'interface"""
        self.log("Interface prête!")
        self.check_api()  # Vérification initiale
        self.root.mainloop()

if __name__ == "__main__":
    # Configuration de l'environnement
    os.environ['VRM_API_BASE'] = 'http://localhost:5030'
    
    print("🚀 Lancement VRAMancer Tkinter GUI")
    
    try:
        app = VRAMancerTkGUI()
        app.run()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        messagebox.showerror("Erreur", f"Impossible de lancer l'interface: {e}")