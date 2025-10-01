import sys
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QProcess

class VRAMancerTray:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.tray = QSystemTrayIcon(QIcon("vramancer.png"), self.app)
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

    def launch_installer(self):
        # Lance l'installateur graphique
        QProcess.startDetached(sys.executable, ["installer_gui.py"])

    def launch_supervision(self):
        # Lance le dashboard supervision (exemple)
        QProcess.startDetached(sys.executable, ["dashboard/dashboard_web.py"])

    def launch_gui(self):
        # Lance la GUI avancée (exemple)
        QProcess.startDetached(sys.executable, ["dashboard/dashboard_qt.py"])

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    tray = VRAMancerTray()
    tray.run()
