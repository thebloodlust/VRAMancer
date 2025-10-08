#!/usr/bin/env python3
"""Vérification et lancement automatique de l'API VRAMancer."""

import requests
import subprocess
import sys
import time
import os

def check_api():
    """Vérifie si l'API VRAMancer est accessible."""
    try:
        response = requests.get("http://localhost:5030/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def main():
    print("=" * 60)
    print("    VRAMANCER - VÉRIFICATION API")
    print("=" * 60)
    print()
    
    print("🔍 Vérification API VRAMancer sur port 5030...")
    
    if check_api():
        print("✅ API VRAMancer ACTIVE")
        print("🚀 Toutes les interfaces sont disponibles")
        print()
        print("Interfaces recommandées:")
        print("  10. Debug Web Ultra (interface web)")
        print("  11. Qt Dashboard (interface native)")
        print("  12. Dashboard Web Avancé")
        print("  13. Mobile Dashboard")
        print()
        return True
    else:
        print("❌ API VRAMancer NON ACCESSIBLE")
        print()
        print("🔧 SOLUTION: Lancez l'API permanente")
        print()
        
        # Vérification fichier API
        if os.path.exists("api_permanente.bat"):
            print("📋 Fichier trouvé: api_permanente.bat")
            
            choice = input("🚀 Lancer l'API automatiquement? (o/N): ")
            if choice.lower() in ['o', 'oui', 'y', 'yes']:
                print()
                print("🔄 Lancement de l'API permanente...")
                print("⚠️  Une nouvelle fenêtre va s'ouvrir - GARDEZ-LA OUVERTE")
                print()
                
                try:
                    if os.name == 'nt':  # Windows
                        subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k', 'api_permanente.bat'])
                    else:  # Linux/Mac
                        subprocess.Popen(['bash', 'api_permanente.bat'])
                    
                    print("✅ API lancée en arrière-plan")
                    print("⏳ Attente démarrage (10 secondes)...")
                    
                    for i in range(10, 0, -1):
                        print(f"   {i}s...", end='\r')
                        time.sleep(1)
                    print()
                    
                    # Re-vérification
                    if check_api():
                        print("✅ API maintenant ACTIVE!")
                        print("🚀 Vous pouvez utiliser toutes les interfaces")
                        return True
                    else:
                        print("⚠️  API pas encore prête")
                        print("   Attendez quelques secondes de plus")
                        return False
                        
                except Exception as e:
                    print(f"❌ Erreur lancement: {e}")
                    return False
            else:
                print()
                print("📋 Instructions manuelles:")
                print("   1. Ouvrez un terminal/cmd")
                print("   2. Exécutez: api_permanente.bat")
                print("   3. Gardez cette fenêtre ouverte")
                print("   4. Relancez ce script pour vérifier")
                return False
        else:
            print("❌ Fichier api_permanente.bat non trouvé")
            print("   Vérifiez que vous êtes dans le bon répertoire")
            return False

if __name__ == "__main__":
    main()