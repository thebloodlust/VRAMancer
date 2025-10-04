import sys, os
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QProcess, Qt

class VRAMancerTray:
    def __init__(self):
        self.app = QApplication(sys.argv)
        # Préserver l'appli si aucune fenêtre
        self.app.setQuitOnLastWindowClosed(False)
        icon = self._resolve_icon()
        self.tray = QSystemTrayIcon(icon, self.app)
        if not QSystemTrayIcon.isSystemTrayAvailable():
            # Log simple sur stdout (Windows: Powershell / cmd)
            print("[Systray] System tray non disponible sur ce système / session (RDP ?)")
        self.menu = QMenu()

        self.install_action = QAction("Installation graphique VRAMancer")
        self.install_action.triggered.connect(self.launch_installer)
        self.menu.addAction(self.install_action)

        self.supervision_action = QAction("Supervision / Dashboard")
        self.supervision_action.triggered.connect(self.launch_supervision)
        self.menu.addAction(self.supervision_action)

        self.gui_qt_action = QAction("Dashboard Qt")
        self.gui_qt_action.triggered.connect(self.launch_gui_qt)
        self.menu.addAction(self.gui_qt_action)

        self.gui_web_action = QAction("Dashboard Web")
        self.gui_web_action.triggered.connect(self.launch_gui_web)
        self.menu.addAction(self.gui_web_action)

        mem_menu = self.menu.addMenu("Mémoire")
        self.promote_action = QAction("Promouvoir 1er bloc")
        self.promote_action.triggered.connect(lambda: self.call_memory_endpoint('promote'))
        mem_menu.addAction(self.promote_action)
        self.demote_action = QAction("Démonter 1er bloc")
        self.demote_action.triggered.connect(lambda: self.call_memory_endpoint('demote'))
        mem_menu.addAction(self.demote_action)

        cli_menu = self.menu.addMenu("CLI")
        self.cli_health = QAction("Healthcheck")
        self.cli_health.triggered.connect(lambda: QProcess.startDetached(sys.executable, ['-m','core.health']))
        cli_menu.addAction(self.cli_health)
        self.cli_list = QAction("Lister GPUs")
        self.cli_list.triggered.connect(
            lambda: QProcess.startDetached(
                sys.executable,
                [
                    '-c',
                    'import sys,os; sys.path.insert(0, os.path.dirname(__file__)); '
                    'from utils.helpers import get_available_gpus; print(get_available_gpus())'
                ]
            )
        )
        cli_menu.addAction(self.cli_list)

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

    def launch_gui_qt(self):
        QProcess.startDetached(sys.executable, ["dashboard/dashboard_qt.py"])

    def launch_gui_web(self):
        QProcess.startDetached(sys.executable, ["dashboard/dashboard_web.py"])

    def call_memory_endpoint(self, action):
        # Appelle l'API /api/memory pour un bloc arbitraire (premier) promote/demote
        import urllib.request, json
        try:
            data = json.loads(urllib.request.urlopen('http://localhost:5000/api/memory').read().decode())
            blocks = list(data.get('blocks', {}).keys())
            if not blocks:
                return
            first = blocks[0][:8]
            if action == 'promote':
                urllib.request.urlopen(f'http://localhost:5000/api/memory/promote?id={first}')
            else:
                urllib.request.urlopen(f'http://localhost:5000/api/memory/demote?id={first}')
        except Exception:
            pass

    def run(self):
        sys.exit(self.app.exec_())

    # ------------------------------------------------------------------
    # Résolution robuste de l'icône (Windows: chemins dupliqués / double extraction)
    # ------------------------------------------------------------------
    def _resolve_icon(self) -> QIcon:
        candidates = [
            os.path.join(os.path.dirname(__file__), "vramancer.png"),
            os.path.join(os.getcwd(), "vramancer.png"),
            os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "vramancer.png"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return QIcon(c)
        # Fallback: icône bleue simple
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.blue)
        return QIcon(pixmap)

if __name__ == "__main__":
    tray = VRAMancerTray()
    tray.run()
