#!/usr/bin/env python3
"""System Tray VRAMancer - Version simplifiée."""

import sys
import os
import time

def main():
    print("=" * 60)
    print("         VRAMANCER SYSTEM TRAY")
    print("=" * 60)
    print()
    
    try:
        # Test import Qt
        try:
            from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction
            from PyQt5.QtGui import QIcon, QPixmap
            from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
            qt_available = True
            qt_version = "PyQt5"
            print("✓ PyQt5 détecté")
        except ImportError:
            try:
                from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction
                from PyQt6.QtGui import QIcon, QPixmap
                from PyQt6.QtCore import QTimer, pyqtSignal, QObject, Qt
                qt_available = True
                qt_version = "PyQt6"
                print("✓ PyQt6 détecté")
            except ImportError:
                qt_available = False
                qt_version = None
                print("❌ Aucune librairie Qt trouvée")
        
        if not qt_available:
            print("Pour utiliser System Tray, installez:")
            print("pip install PyQt5")
            print("\nSimulation du System Tray...")
            for i in range(10):
                print(f"[{time.strftime('%H:%M:%S')}] System Tray actif - Cycle {i+1}/10")
                time.sleep(2)
            print("System Tray arrêté")
            return
        
        # Test si système supporte system tray
        app = QApplication(sys.argv)
        if not QSystemTrayIcon.isSystemTrayAvailable():
            print("❌ System Tray non supporté sur ce système")
            print("Simulation du monitoring...")
            for i in range(5):
                print(f"[{time.strftime('%H:%M:%S')}] Monitoring - Cycle {i+1}/5")
                time.sleep(2)
            return
        
        print("✓ System Tray supporté")
        print("Démarrage du System Tray...")
        
        # Création du System Tray avec icône VRAMancer
        tray_icon = QSystemTrayIcon()
        
        # Recherche et chargement de l'icône vramancer.png
        icon_paths = [
            os.path.join(os.path.dirname(__file__), "vramancer.png"),
            os.path.join(os.getcwd(), "vramancer.png"),
            "vramancer.png"
        ]
        
        icon_loaded = False
        for icon_path in icon_paths:
            if os.path.exists(icon_path):
                tray_icon.setIcon(QIcon(icon_path))
                print(f"✓ Icône VRAMancer chargée: {icon_path}")
                icon_loaded = True
                break
        
        if not icon_loaded:
            # Icône de fallback bleue
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.blue)
            tray_icon.setIcon(QIcon(pixmap))
            print("⚠️  Icône par défaut utilisée (vramancer.png non trouvé)")
        
        # Menu du tray
        menu = QMenu()
        
        action_status = QAction("VRAMancer - Actif")
        action_status.setEnabled(False)
        menu.addAction(action_status)
        
        menu.addSeparator()
        
        action_dashboard = QAction("Ouvrir Dashboard")
        menu.addAction(action_dashboard)
        
        action_quit = QAction("Quitter")
        action_quit.triggered.connect(app.quit)
        menu.addAction(action_quit)
        
        tray_icon.setContextMenu(menu)
        tray_icon.setToolTip("VRAMancer System Tray")
        
        # Affichage
        tray_icon.show()
        
        print("✓ System Tray démarré")
        print("Regardez dans la barre des tâches pour l'icône VRAMancer")
        print("Clic droit sur l'icône pour accéder au menu")
        print("\nLe System Tray fonctionne en arrière-plan...")
        print("Fermeture automatique dans 30 secondes...")
        
        # Timer pour fermer automatiquement après 30s
        timer = QTimer()
        timer.timeout.connect(app.quit)
        timer.start(30000)  # 30 secondes
        
        # Lancement de l'application
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"❌ Erreur System Tray: {e}")
        print("Mode simulation...")
        for i in range(5):
            print(f"[{time.strftime('%H:%M:%S')}] Simulation System Tray - {i+1}/5")
            time.sleep(1)

if __name__ == "__main__":
    main()