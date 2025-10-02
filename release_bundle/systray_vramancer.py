import sys
import os
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QProcess

class VRAMancerTray:
    def __init__(self):
        print("[VRAMancer Systray] Démarrage du systray...")
        self.app = QApplication(sys.argv)
        # Détection automatique du chemin du bundle, même si dans un sous-dossier du dépôt
        import sys
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # Si le script est dans release_bundle, on garde ce chemin
        # Sinon, on cherche le dossier release_bundle dans les parents
        bundle_dir = self.base_dir
        while not os.path.basename(bundle_dir).startswith("release_bundle"):
            parent = os.path.dirname(bundle_dir)
            if parent == bundle_dir:
                break
            bundle_dir = parent
        self.bundle_dir = bundle_dir
        print(f"[Systray] Dossier bundle détecté : {self.bundle_dir}")
        # Ajout des chemins core et dashboard au sys.path pour les imports
        core_path = os.path.join(self.bundle_dir, "core")
        dashboard_path = os.path.join(self.bundle_dir, "dashboard")
        if core_path not in sys.path:
            sys.path.append(core_path)
        if dashboard_path not in sys.path:
            sys.path.append(dashboard_path)
        icon_path = os.path.join(self.bundle_dir, "vramancer.png")
        if not os.path.exists(icon_path):
            print(f"[ERREUR] Icône non trouvée : {icon_path}")
        try:
            icon = QIcon(icon_path)
            if icon.isNull():
                print(f"[ERREUR] Impossible de charger l'icône : {icon_path}")
        except Exception as e:
            print(f"[ERREUR] Exception lors du chargement de l'icône : {e}")
            icon = QIcon()
        self.tray = QSystemTrayIcon(icon, self.app)
        self.menu = QMenu()

        self.install_action = QAction("Installation graphique VRAMancer")
        self.install_action.triggered.connect(self.launch_installer)
        self.menu.addAction(self.install_action)

        self.supervision_action = QAction("Supervision / Dashboard")
        self.supervision_action.triggered.connect(self.launch_supervision)
        self.menu.addAction(self.supervision_action)

        self.gui_action = QAction("Ouvrir GUI avancée")
        self.gui_action.triggered.connect(self.launch_gui)
        self.menu.addAction(self.gui_action)

        self.quit_action = QAction("Quitter")
        self.quit_action.triggered.connect(self.app.quit)
        self.menu.addAction(self.quit_action)

        self.tray.setContextMenu(self.menu)
        self.tray.setToolTip("VRAMancer est lancé")
        self.tray.show()
        print("[VRAMancer Systray] Systray lancé. Vérifiez la barre de tâches.")

    def launch_installer(self):
        # Lance l'installateur graphique
        installer_path = os.path.join(self.bundle_dir, "installer_gui.py")
        if not os.path.exists(installer_path):
            print(f"[ERREUR] Installateur non trouvé : {installer_path}")
        QProcess.startDetached(sys.executable, [installer_path])

    def launch_supervision(self):
        # Lance le dashboard supervision (exemple)
        dashboard_path = os.path.join(self.bundle_dir, "dashboard", "dashboard_web.py")
        if not os.path.exists(dashboard_path):
            print(f"[ERREUR] Dashboard non trouvé : {dashboard_path}")
        QProcess.startDetached(sys.executable, [dashboard_path])

    def launch_gui(self):
        # Lance la GUI avancée (exemple)
        gui_path = os.path.join(self.bundle_dir, "dashboard", "dashboard_qt.py")
        if not os.path.exists(gui_path):
            print(f"[ERREUR] GUI avancée non trouvée : {gui_path}")
        QProcess.startDetached(sys.executable, [gui_path])

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    tray = VRAMancerTray()
    tray.run()
