#!/usr/bin/env python3
"""Dashboard CLI VRAMancer."""

import sys
import os
import requests
import json
import time
from datetime import datetime

def launch():
    """Lance le dashboard CLI VRAMancer."""
    print("=" * 80)
    print("                    VRAMANCER CLI DASHBOARD")
    print("=" * 80)
    print()
    
    api_url = "http://localhost:5030"
    
    while True:
        try:
            # Test connexion API
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ API VRAMancer connectee ({datetime.now().strftime('%H:%M:%S')})")
                
                # Infos GPU
                try:
                    gpu_response = requests.get(f"{api_url}/api/gpu/info", timeout=5)
                    if gpu_response.status_code == 200:
                        gpu_data = gpu_response.json()
                        print(f"📊 GPUs detectes: {len(gpu_data.get('gpus', []))}")
                        for gpu in gpu_data.get('gpus', []):
                            print(f"   - {gpu.get('name', 'GPU')} ({gpu.get('memory_used', 0)}MB/{gpu.get('memory_total', 0)}MB)")
                except:
                    print("⚠️  Impossible de recuperer les infos GPU")
                
                # Status systeme
                try:
                    status_response = requests.get(f"{api_url}/api/status", timeout=5)
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        print(f"🔧 Status: {status_data.get('status', 'inconnu')}")
                        print(f"⏱️  Uptime: {status_data.get('uptime', 0):.1f}s")
                except:
                    print("⚠️  Impossible de recuperer le status")
                    
            else:
                print(f"❌ API VRAMancer non accessible (Status: {response.status_code})")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur connexion API: {str(e)}")
            print("   Verifiez que l'API VRAMancer fonctionne (option 1 du menu)")
        
        print("\n" + "-" * 80)
        print("Commandes: [R]afraichir, [Q]uitter")
        
        try:
            choice = input("Choix: ").upper()
            if choice == 'Q':
                break
            elif choice == 'R':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
        except KeyboardInterrupt:
            break
            
        time.sleep(5)
        os.system('cls' if os.name == 'nt' else 'clear')
    
    print("Dashboard CLI ferme")

if __name__ == "__main__":
    launch()
