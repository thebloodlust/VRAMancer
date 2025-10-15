#!/usr/bin/env python3
"""
VRAMancer API - Point d'entrée principal
Wrapper qui utilise la version production avec fallback vers simple si nécessaire
"""
import os
import sys

# Déterminer le mode d'exécution
PRODUCTION_MODE = os.environ.get('VRM_PRODUCTION', '1') in {'1', 'true', 'TRUE'}

if PRODUCTION_MODE:
    # Mode production : utiliser l'API robuste
    try:
        from core.production_api import main
        
        if __name__ == '__main__':
            main()
    except ImportError as e:
        print(f"⚠️  Impossible de charger l'API production: {e}")
        print("   Fallback vers api_simple.py...")
        PRODUCTION_MODE = False

if not PRODUCTION_MODE or 'main' not in dir():
    # Mode simple : utiliser l'ancienne version (dev/debug seulement)
    print("⚠️  Mode SIMPLE actif - Ne PAS utiliser en production!")
    print("   Pour mode production: export VRM_PRODUCTION=1")
    
    # Importer l'ancienne version
    exec(open('api_simple.py').read())
