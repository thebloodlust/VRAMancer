#!/usr/bin/env python3
"""CLI Dashboard VRAMancer - Version non-bloquante."""

import sys
import os
import requests
import json
import time
from datetime import datetime

def test_api():
    """Test si l'API est accessible."""
    api_url = "http://localhost:5030"
    try:
        response = requests.get(f"{api_url}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def get_gpu_info():
    """Récupère les infos GPU."""
    api_url = "http://localhost:5030"
    try:
        response = requests.get(f"{api_url}/api/gpu/info", timeout=3)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_system_status():
    """Récupère le status système."""
    api_url = "http://localhost:5030"
    try:
        response = requests.get(f"{api_url}/api/status", timeout=3)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def main():
    print("=" * 80)
    print("                    VRAMANCER CLI DASHBOARD")
    print("=" * 80)
    print()
    
    # Test API
    if test_api():
        print(f"✓ API VRAMancer connectée ({datetime.now().strftime('%H:%M:%S')})")
        
        # Infos GPU
        gpu_data = get_gpu_info()
        if gpu_data:
            gpus = gpu_data.get('gpus', [])
            print(f"📊 GPUs détectés: {len(gpus)}")
            for i, gpu in enumerate(gpus[:3]):  # Limite à 3 pour l'affichage
                name = gpu.get('name', 'GPU Inconnu')
                memory = f"{gpu.get('memory_used', 0)}MB/{gpu.get('memory_total', 0)}MB"
                print(f"   - GPU {i+1}: {name} ({memory})")
        else:
            print("⚠️  Impossible de récupérer les infos GPU")
        
        # Status système
        status_data = get_system_status()
        if status_data:
            print(f"🔧 Status: {status_data.get('status', 'inconnu')}")
            print(f"⏱️  Uptime: {status_data.get('uptime', 0):.1f}s")
        else:
            print("⚠️  Impossible de récupérer le status")
            
    else:
        print("❌ API VRAMancer non accessible")
        print("   Vérifiez que l'API VRAMancer fonctionne (option 1 du menu)")
        print("   Ou lancez: api_permanente.bat")
    
    print()
    print("=" * 80)
    print("CLI Dashboard - Affichage terminé")
    print("Pour une interface interactive complète, utilisez l'option 11 (Qt Dashboard)")
    print("=" * 80)

if __name__ == "__main__":
    main()