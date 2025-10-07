#!/usr/bin/env python3
"""Dashboard Tkinter VRAMancer."""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import requests
import json
import threading
import time
import sys
from datetime import datetime

class VRAMancerTkinterDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VRAMancer Dashboard - Tkinter")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        self.api_url = "http://localhost:5030"
        self.running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Configure l'interface utilisateur."""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Titre
        title_label = ttk.Label(main_frame, text="VRAMancer Dashboard", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Frame pour boutons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(button_frame, text="Demarrer Monitoring", 
                                      command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="Arreter Monitoring", 
                                     command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.refresh_button = ttk.Button(button_frame, text="Rafraichir", 
                                        command=self.manual_refresh)
        self.refresh_button.pack(side=tk.LEFT)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Status: Arrete", 
                                     foreground="red")
        self.status_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Notebook pour onglets
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Onglet GPU Info
        self.gpu_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.gpu_frame, text="GPU Info")
        
        self.gpu_text = scrolledtext.ScrolledText(self.gpu_frame, wrap=tk.WORD)
        self.gpu_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Onglet System Status
        self.status_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.status_frame, text="System Status")
        
        self.status_text = scrolledtext.ScrolledText(self.status_frame, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Onglet Logs
        self.logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_frame, text="Logs")
        
        self.logs_text = scrolledtext.ScrolledText(self.logs_frame, wrap=tk.WORD)
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initial refresh
        self.manual_refresh()
        
    def log_message(self, message):
        """Ajoute un message aux logs."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.logs_text.see(tk.END)
        
    def start_monitoring(self):
        """Demarre le monitoring automatique."""
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Actif", foreground="green")
        
        self.log_message("Monitoring demarre")
        
        # Thread de monitoring
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Arrete le monitoring automatique."""
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Arrete", foreground="red")
        
        self.log_message("Monitoring arrete")
        
    def monitor_loop(self):
        """Boucle de monitoring en arriere-plan."""
        while self.running:
            self.manual_refresh()
            time.sleep(5)  # Rafraichir toutes les 5 secondes
            
    def manual_refresh(self):
        """Rafraichissement manuel des donnees."""
        self.log_message("Rafraichissement des donnees...")
        
        # Test API
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                self.log_message("✓ API VRAMancer accessible")
                self.refresh_gpu_info()
                self.refresh_system_status()
            else:
                self.log_message(f"❌ API VRAMancer erreur: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_message(f"❌ Erreur connexion API: {str(e)}")
            
    def refresh_gpu_info(self):
        """Rafraichit les informations GPU."""
        try:
            response = requests.get(f"{self.api_url}/api/gpu/info", timeout=5)
            if response.status_code == 200:
                gpu_data = response.json()
                
                self.gpu_text.delete(1.0, tk.END)
                self.gpu_text.insert(tk.END, "=== INFORMATIONS GPU ===\n\n")
                
                gpus = gpu_data.get('gpus', [])
                if gpus:
                    for i, gpu in enumerate(gpus):
                        self.gpu_text.insert(tk.END, f"GPU {i+1}:\n")
                        self.gpu_text.insert(tk.END, f"  Nom: {gpu.get('name', 'Inconnu')}\n")
                        self.gpu_text.insert(tk.END, f"  Memoire: {gpu.get('memory_used', 0)}MB / {gpu.get('memory_total', 0)}MB\n")
                        self.gpu_text.insert(tk.END, f"  Utilisation: {gpu.get('utilization', 0)}%\n")
                        self.gpu_text.insert(tk.END, f"  Temperature: {gpu.get('temperature', 0)}°C\n\n")
                else:
                    self.gpu_text.insert(tk.END, "Aucun GPU detecte\n")
                    
        except Exception as e:
            self.gpu_text.delete(1.0, tk.END)
            self.gpu_text.insert(tk.END, f"Erreur lors de la recuperation des infos GPU: {str(e)}\n")
            
    def refresh_system_status(self):
        """Rafraichit le status systeme."""
        try:
            response = requests.get(f"{self.api_url}/api/status", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                
                self.status_text.delete(1.0, tk.END)
                self.status_text.insert(tk.END, "=== STATUS SYSTEME ===\n\n")
                self.status_text.insert(tk.END, f"Status: {status_data.get('status', 'inconnu')}\n")
                self.status_text.insert(tk.END, f"Uptime: {status_data.get('uptime', 0):.1f} secondes\n")
                self.status_text.insert(tk.END, f"Version: {status_data.get('version', 'inconnue')}\n")
                self.status_text.insert(tk.END, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
        except Exception as e:
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(tk.END, f"Erreur lors de la recuperation du status: {str(e)}\n")
            
    def run(self):
        """Lance l'interface Tkinter."""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except Exception as e:
            print(f"Erreur Tkinter: {e}")
            
    def on_closing(self):
        """Gestion de la fermeture de la fenetre."""
        if self.running:
            self.stop_monitoring()
        self.root.destroy()

def launch():
    """Lance le dashboard Tkinter."""
    try:
        dashboard = VRAMancerTkinterDashboard()
        dashboard.run()
    except Exception as e:
        print(f"Erreur lors du lancement du dashboard Tkinter: {e}")
        sys.exit(1)

def launch_dashboard():
    """Alias pour compatibilite.""" 
    launch()

if __name__ == "__main__":
    launch()
