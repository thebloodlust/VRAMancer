#!/usr/bin/env python3
"""Test des liens System Tray sans interface graphique."""

import os
import sys

def test_systray_links():
    print("=" * 60)
    print("    TEST LIENS SYSTEM TRAY VRAMANCER")
    print("=" * 60)
    print()
    
    # Test 1: Fichiers de lancement
    files_to_test = [
        ("Qt Dashboard", "dashboard/dashboard_qt.py"),
        ("Debug Web Ultra", "debug_web_ultra.py"),
        ("Dashboard Web Avancé", "dashboard/dashboard_web_advanced.py"),
        ("Dashboard Web Standard", "dashboard/dashboard_web.py"),
        ("Mobile Dashboard", "mobile/dashboard_mobile.py"),
        ("Installateur", "installer_gui.py"),
    ]
    
    print("🔍 Vérification fichiers disponibles:")
    available_files = []
    
    for name, path in files_to_test:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✅ {name}: {path} ({size} bytes)")
            available_files.append((name, path))
        else:
            print(f"  ❌ {name}: {path} (absent)")
    
    print()
    
    # Test 2: Simulation lancement (sans vraiment lancer)
    print("🚀 Test simulation lancement:")
    for name, path in available_files:
        try:
            # Test si le fichier est Python valide
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'if __name__ == "__main__"' in content or 'def main(' in content:
                    print(f"  ✅ {name}: Prêt à lancer")
                else:
                    print(f"  ⚠️  {name}: Pas de point d'entrée détecté")
        except Exception as e:
            print(f"  ❌ {name}: Erreur lecture ({e})")
    
    print()
    
    # Test 3: API endpoint pour actions mémoire
    print("🔗 Test connectivité API (actions mémoire):")
    try:
        import requests
        response = requests.get("http://localhost:5030/health", timeout=3)
        if response.status_code == 200:
            print("  ✅ API accessible sur port 5030")
            
            # Test endpoint mémoire
            try:
                mem_response = requests.get("http://localhost:5030/api/memory", timeout=3)
                print(f"  ✅ Endpoint mémoire: Status {mem_response.status_code}")
            except:
                print("  ⚠️  Endpoint mémoire non disponible (normal si pas d'API complète)")
        else:
            print(f"  ⚠️  API répond avec status {response.status_code}")
    except:
        print("  ❌ API non accessible - Lancez api_permanente.bat d'abord")
    
    print()
    
    # Résumé
    print("=" * 60)
    print("RÉSUMÉ:")
    print(f"  📊 {len(available_files)}/{len(files_to_test)} fichiers disponibles")
    
    if len(available_files) >= 4:
        print("  ✅ System Tray aura suffisamment d'options")
    else:
        print("  ⚠️  Peu d'options disponibles pour System Tray")
    
    print()
    print("RECOMMANDATIONS:")
    print("  1. Lancez d'abord: api_permanente.bat")
    print("  2. Puis lancez: python systray_vramancer.py")
    print("  3. Clic droit sur l'icône → Testez les liens")
    print("=" * 60)

if __name__ == "__main__":
    test_systray_links()