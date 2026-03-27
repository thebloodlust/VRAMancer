#!/usr/bin/env python3
"""Dashboard CLI VRAMancer."""

import sys
import os
import subprocess
import requests
import json
import time
from datetime import datetime

def clear_console():
    subprocess.run(['cls' if os.name == 'nt' else 'clear'], shell=True)

def launch():
    """Lance le dashboard CLI VRAMancer."""
    print("=" * 80)
    print("                    VRAMANCER CLI DASHBOARD")
    print("=" * 80)
    print()
    
    port = os.environ.get("VRM_API_PORT", "5030")
    token = os.environ.get("VRM_API_TOKEN", "")
    api_url = f"http://localhost:{port}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    while True:
        try:
            # Test connexion API
            response = requests.get(f"{api_url}/health", headers=headers, timeout=5)
            if response.status_code == 200:
                health_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                print(f" API VRAMancer connectee ({datetime.now().strftime('%H:%M:%S')})")
                
                # Infos GPU — /api/gpu returns {cuda_available, device_count, devices: [...]}
                try:
                    gpu_response = requests.get(f"{api_url}/api/gpu", headers=headers, timeout=5)
                    if gpu_response.status_code == 200:
                        gpu_data = gpu_response.json()
                        devices = gpu_data.get('devices', [])
                        print(f" GPUs detectes: {len(devices)}")
                        for gpu in devices:
                            used_mb = int(gpu.get('memory_used', 0) / (1024**2))
                            total_mb = int(gpu.get('memory_total', 0) / (1024**2))
                            usage = gpu.get('memory_usage_percent', 0)
                            print(f"   - {gpu.get('name', 'GPU')} ({used_mb}MB/{total_mb}MB, {usage}%)")
                except Exception:
                    print("  Impossible de recuperer les infos GPU")
                
                # Status systeme — /api/status returns {backend, version, endpoints, ...}
                try:
                    status_response = requests.get(f"{api_url}/api/status", headers=headers, timeout=5)
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        print(f" Backend: {status_data.get('backend', 'inconnu')}")
                        print(f" Version: {status_data.get('version', '?')}")
                except Exception:
                    print("  Impossible de recuperer le status")

                # Pipeline status — /api/pipeline/status
                try:
                    pipe_response = requests.get(f"{api_url}/api/pipeline/status", headers=headers, timeout=5)
                    if pipe_response.status_code == 200:
                        pipe_data = pipe_response.json()
                        model = pipe_data.get('model') or '(aucun)'
                        n_gpus = pipe_data.get('num_gpus', 0)
                        mode = pipe_data.get('parallel_mode', 'pp')
                        print(f" Modele: {model} ({n_gpus} GPU, mode={mode})")
                except Exception:
                    pass
                    
            else:
                print(f" API VRAMancer non accessible (Status: {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f" Erreur connexion API: {str(e)}")
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
