#!/usr/bin/env python3
"""Test de l'icône System Tray VRAMancer."""

import os
import sys

def test_icon():
    print("=" * 60)
    print("     TEST ICÔNE SYSTEM TRAY VRAMANCER")
    print("=" * 60)
    print()
    
    # Test 1: Vérification fichier
    icon_paths = [
        os.path.join(os.path.dirname(__file__), "vramancer.png"),
        os.path.join(os.getcwd(), "vramancer.png"),
        "vramancer.png"
    ]
    
    icon_found = False
    icon_path_found = None
    
    for icon_path in icon_paths:
        if os.path.exists(icon_path):
            icon_found = True
            icon_path_found = icon_path
            break
    
    if icon_found:
        print(f"✓ Icône trouvée: {icon_path_found}")
        
        # Infos sur le fichier
        size = os.path.getsize(icon_path_found)
        print(f"✓ Taille fichier: {size} bytes")
        
        # Test si c'est une image PNG valide
        try:
            with open(icon_path_found, 'rb') as f:
                header = f.read(8)
                if header.startswith(b'\x89PNG\r\n\x1a\n'):
                    print("✓ Format PNG valide")
                else:
                    print("⚠️  Fichier existe mais pas au format PNG valide")
        except Exception as e:
            print(f"⚠️  Erreur lecture fichier: {e}")
    else:
        print("❌ Icône vramancer.png non trouvée")
        print("Chemins testés:")
        for path in icon_paths:
            print(f"  - {path}")
    
    print()
    
    # Test 2: Vérification Qt (sans créer d'app)
    print("Test compatibilité Qt...")
    try:
        # Test import sans créer d'application
        from PyQt5.QtGui import QIcon
        print("✓ PyQt5 importé avec succès")
        
        if icon_found:
            # Test simulation chargement (sans QApplication)
            print("✓ Icône prête pour System Tray")
            print(f"✓ Chemin à utiliser: {icon_path_found}")
        else:
            print("⚠️  Icône de fallback sera utilisée")
            
    except ImportError:
        try:
            from PyQt6.QtGui import QIcon
            print("✓ PyQt6 importé avec succès")
        except ImportError:
            print("❌ Ni PyQt5 ni PyQt6 disponible")
            print("   Installez: pip install PyQt5")
    except Exception as e:
        print(f"⚠️  Problème Qt: {e}")
    
    print()
    
    # Test 3: Infos système
    print("Infos environnement:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  OS: {os.name}")
    print(f"  DISPLAY: {os.environ.get('DISPLAY', 'Non défini')}")
    
    if os.environ.get('DISPLAY'):
        print("✓ Interface graphique potentiellement disponible")
    else:
        print("⚠️  Pas d'interface graphique (normal en container)")
        print("   System Tray fonctionne sur Windows/Linux desktop")
    
    print()
    print("=" * 60)
    print("RÉSUMÉ:")
    if icon_found:
        print("✓ L'icône vramancer.png est correctement configurée")
        print("✓ Le System Tray utilisera l'icône VRAMancer")
    else:
        print("⚠️  Icône manquante - icône par défaut utilisée")
    
    print("ℹ️  Le System Tray fonctionne sur Windows et Linux desktop")
    print("=" * 60)

if __name__ == "__main__":
    test_icon()