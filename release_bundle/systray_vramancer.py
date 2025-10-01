import sys
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QProcess

class VRAMancerTray:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.tray = QSystemTrayIcon(QIcon("vramancer.png"), self.app)
        self.menu = QMenu()

        self.launch_action = QAction("Ouvrir l'interface VRAMancer")
        self.launch_action.triggered.connect(self.launch_gui)
        self.menu.addAction(self.launch_action)

        self.quit_action = QAction("Quitter")
        self.quit_action.triggered.connect(self.app.quit)
        self.menu.addAction(self.quit_action)

        self.tray.setContextMenu(self.menu)
        self.tray.setToolTip("VRAMancer est lancé")
        self.tray.show()

    def launch_gui(self):
        # Lance l'interface graphique principale
        QProcess.startDetached(sys.executable, ["installer_gui.py"])

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    tray = VRAMancerTray()
    tray.run()
