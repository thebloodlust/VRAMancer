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
            from PyQt5.QtGui import QIcon
            from PyQt5.QtCore import QTimer, pyqtSignal, QObject
            qt_available = True
            print("✓ PyQt5 détecté")
        except ImportError:
            try:
                from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction
                from PyQt6.QtGui import QIcon
                from PyQt6.QtCore import QTimer, pyqtSignal, QObject
                qt_available = True
                print("✓ PyQt6 détecté")
            except ImportError:
                qt_available = False
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
        
        # Création du System Tray
        tray_icon = QSystemTrayIcon()
        
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
        
        # Icône par défaut (si pas d'icône spécifique)
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