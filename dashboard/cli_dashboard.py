#!/usr/bin/env python3
"""Dashboard CLI VRAMancer."""

import sys
import os
import requests
import json
import time
from datetime import datetime

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def launch():
    """Lance le dashboard CLI VRAMancer."""
    print("=" * 80)
    print("                    VRAMANCER CLI DASHBOARD")
    print("=" * 80)
    print()
    
    port = os.environ.get("VRM_API_PORT", "8000")
    api_url = f"http://localhost:{port}"
    
    while True:
        try:
            # Test connexion API
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ API VRAMancer connectee ({datetime.now().strftime('%H:%M:%S')})")
                
                # Infos GPU
                try:
                    gpu_response = requests.get(f"{api_url}/api/gpu", timeout=5)
                    if gpu_response.status_code == 200:
                        gpu_data = gpu_response.json()
                        print(f"📊 GPUs detectes: {len(gpu_data.get('devices', []))}")
                        for gpu in gpu_data.get('devices', []):
                            used_mb = int(gpu.get('memory_used', 0) / (1024**2))
                            total_mb = int(gpu.get('memory_total', 0) / (1024**2))
                            print(f"   - {gpu.get('name', 'GPU')} ({used_mb}MB/{total_mb}MB)")
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
                clear_console()
                continue
        except KeyboardInterrupt:
            break
            
        time.sleep(5)
        clear_console()
    
    print("Dashboard CLI ferme")

if __name__ == "__main__":
    launch()
