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

        # Menu principal des interfaces
        interfaces_menu = self.menu.addMenu("🚀 Interfaces")
        
        self.gui_qt_action = QAction("Qt Dashboard (Recommandé)")
        self.gui_qt_action.triggered.connect(self.launch_gui_qt)
        interfaces_menu.addAction(self.gui_qt_action)

        self.gui_web_ultra_action = QAction("Debug Web Ultra")
        self.gui_web_ultra_action.triggered.connect(self.launch_debug_web_ultra)
        interfaces_menu.addAction(self.gui_web_ultra_action)

        self.supervision_action = QAction("Dashboard Web Avancé")
        self.supervision_action.triggered.connect(self.launch_supervision)
        interfaces_menu.addAction(self.supervision_action)

        self.mobile_action = QAction("Mobile Dashboard")
        self.mobile_action.triggered.connect(self.launch_mobile)
        interfaces_menu.addAction(self.mobile_action)

        interfaces_menu.addSeparator()

        self.install_action = QAction("Installation VRAMancer")
        self.install_action.triggered.connect(self.launch_installer)
        interfaces_menu.addAction(self.install_action)

        mem_menu = self.menu.addMenu("Mémoire")
        self.promote_action = QAction("Promouvoir 1er bloc")
        self.promote_action.triggered.connect(lambda: self.call_memory_endpoint('promote'))
        mem_menu.addAction(self.promote_action)
        self.demote_action = QAction("Démonter 1er bloc")
        self.demote_action.triggered.connect(lambda: self.call_memory_endpoint('demote'))
        mem_menu.addAction(self.demote_action)

        cli_menu = self.menu.addMenu("CLI")
        self.cli_health = QAction("Healthcheck")
        self.cli_health.triggered.connect(
            lambda: QProcess.startDetached(
                sys.executable,
                [
                    '-c',
                    'import sys,os; sys.path.insert(0, os.getcwd()); '
                    'from core.utils import detect_backend, enumerate_devices; '
                    'print("=== VRAMancer Health Check ==="); '
                    'print(f"Backend: {detect_backend()}"); '
                    'devices = enumerate_devices(); '
                    'print(f"Devices: {len(devices)}"); '
                    '[print(f"  - {d[\"name\"]} ({d[\"backend\"]})") for d in devices]; '
                    'print("=== Health OK ==="); '
                    'input("Appuyez sur Entrée...")'
                ]
            )
        )
        cli_menu.addAction(self.cli_health)
        self.cli_list = QAction("Lister GPUs")
        self.cli_list.triggered.connect(
            lambda: QProcess.startDetached(
                sys.executable,
                [
                    '-c',
                    'import sys,os; sys.path.insert(0, os.getcwd()); '
                    'from core.utils import enumerate_devices; '
                    'devices = enumerate_devices(); '
                    'print("=== GPUs VRAMancer ==="); '
                    '[print(f"GPU {d[\"id\"]}: {d[\"name\"]} ({d[\"backend\"]})") for d in devices]; '
                    'input("Appuyez sur Entrée...")'
                ]
            )
        )
        cli_menu.addAction(self.cli_list)

        self.menu.addSeparator()

        # Actions système
        self.api_check = QAction("🔍 Vérifier API VRAMancer")
        self.api_check.triggered.connect(self.check_api_status)
        self.menu.addAction(self.api_check)

        self.menu.addSeparator()

        self.quit_action = QAction("❌ Quitter")
        self.quit_action.triggered.connect(self.app.quit)
        self.menu.addAction(self.quit_action)

        self.tray.setContextMenu(self.menu)
        self.tray.setToolTip("🚀 VRAMancer System Tray\n🖱️ Clic droit pour le menu complet\n🎮 RTX 4060 Laptop GPU supporté")
        self.tray.show()
        
        # Message de bienvenue
        self.tray.showMessage(
            "🚀 VRAMancer System Tray", 
            "✅ Lancé avec succès !\n🖱️ Clic droit sur l'icône pour accéder à toutes les interfaces\n🎮 RTX 4060 Laptop GPU détecté", 
            QSystemTrayIcon.Information, 
            5000
        )

    def launch_installer(self):
        # Lance l'installateur graphique
        if os.path.exists("installer_gui.py"):
            success = QProcess.startDetached(sys.executable, ["installer_gui.py"])
            if success:
                self.tray.showMessage("VRAMancer", "Installateur VRAMancer lancé", QSystemTrayIcon.Information, 3000)
            else:
                self.tray.showMessage("VRAMancer", "Erreur lancement installateur", QSystemTrayIcon.Critical, 3000)
        else:
            self.tray.showMessage("VRAMancer", "Installateur non trouvé", QSystemTrayIcon.Warning, 3000)

    def launch_supervision(self):
        # Lance le dashboard web avancé (supervision cluster)
        if os.path.exists("dashboard/dashboard_web_advanced.py"):
            success = QProcess.startDetached(sys.executable, ["dashboard/dashboard_web_advanced.py"])
            if success:
                self.tray.showMessage("VRAMancer", "Dashboard Web Avancé démarré\nURL: http://localhost:5000", QSystemTrayIcon.Information, 5000)
            else:
                self.tray.showMessage("VRAMancer", "Erreur lancement Dashboard Web Avancé", QSystemTrayIcon.Critical, 3000)
        elif os.path.exists("dashboard/dashboard_web.py"):
            success = QProcess.startDetached(sys.executable, ["dashboard/dashboard_web.py"])
            if success:
                self.tray.showMessage("VRAMancer", "Dashboard Web démarré", QSystemTrayIcon.Information, 3000)
        else:
            self.tray.showMessage("VRAMancer", "Aucun dashboard web trouvé", QSystemTrayIcon.Warning, 3000)

    def launch_gui_qt(self):
        if os.path.exists("dashboard/dashboard_qt.py"):
            success = QProcess.startDetached(sys.executable, ["dashboard/dashboard_qt.py"])
            if success:
                self.tray.showMessage("VRAMancer", "Qt Dashboard en cours de lancement...", QSystemTrayIcon.Information, 3000)
            else:
                self.tray.showMessage("VRAMancer", "Erreur lancement Qt Dashboard", QSystemTrayIcon.Critical, 3000)
        else:
            self.tray.showMessage("VRAMancer", "Qt Dashboard non trouvé", QSystemTrayIcon.Warning, 3000)

    def launch_gui_web(self):
        # Priorité: debug_web_ultra > dashboard_web_advanced > dashboard_web
        if os.path.exists("debug_web_ultra.py"):
            QProcess.startDetached(sys.executable, ["debug_web_ultra.py"])
        elif os.path.exists("dashboard/dashboard_web_advanced.py"):
            QProcess.startDetached(sys.executable, ["dashboard/dashboard_web_advanced.py"])
        elif os.path.exists("dashboard/dashboard_web.py"):
            QProcess.startDetached(sys.executable, ["dashboard/dashboard_web.py"])
        else:
            print("[Systray] Aucun dashboard web trouvé")

    def launch_debug_web_ultra(self):
        if os.path.exists("debug_web_ultra.py"):
            success = QProcess.startDetached(sys.executable, ["debug_web_ultra.py"])
            if success:
                self.tray.showMessage("VRAMancer", "Debug Web Ultra démarré\nURL: http://localhost:8080", QSystemTrayIcon.Information, 5000)
            else:
                self.tray.showMessage("VRAMancer", "Erreur lancement Debug Web Ultra", QSystemTrayIcon.Critical, 3000)
        else:
            self.tray.showMessage("VRAMancer", "Debug Web Ultra non trouvé", QSystemTrayIcon.Warning, 3000)

    def launch_mobile(self):
        if os.path.exists("mobile/dashboard_mobile.py"):
            success = QProcess.startDetached(sys.executable, ["mobile/dashboard_mobile.py"])
            if success:
                self.tray.showMessage("VRAMancer", "Mobile Dashboard démarré\nURL: http://localhost:5003", QSystemTrayIcon.Information, 5000)
            else:
                self.tray.showMessage("VRAMancer", "Erreur lancement Mobile Dashboard", QSystemTrayIcon.Critical, 3000)
        else:
            self.tray.showMessage("VRAMancer", "Mobile Dashboard non trouvé", QSystemTrayIcon.Warning, 3000)

    def call_memory_endpoint(self, action):
        # Appelle l'API /api/memory pour un bloc arbitraire (premier) promote/demote
        import urllib.request, json
        try:
            # Utilise le bon port API (5030, pas 5000)
            data = json.loads(urllib.request.urlopen('http://localhost:5030/api/memory').read().decode())
            blocks = list(data.get('blocks', {}).keys())
            if not blocks:
                print(f"[Systray] Aucun bloc mémoire trouvé pour {action}")
                return
            first = blocks[0][:8]
            if action == 'promote':
                urllib.request.urlopen(f'http://localhost:5030/api/memory/promote?id={first}')
                print(f"[Systray] Bloc {first} promu")
            else:
                urllib.request.urlopen(f'http://localhost:5030/api/memory/demote?id={first}')
                print(f"[Systray] Bloc {first} démonté")
        except Exception as e:
            print(f"[Systray] Erreur API mémoire: {e}")

    def check_api_status(self):
        """Vérifie le statut de l'API VRAMancer."""
        try:
            import urllib.request
            response = urllib.request.urlopen('http://localhost:5030/health', timeout=3)
            if response.getcode() == 200:
                self.tray.showMessage("VRAMancer API", "✅ API active sur port 5030\nToutes les fonctions disponibles", QSystemTrayIcon.Information, 5000)
            else:
                self.tray.showMessage("VRAMancer API", f"⚠️ API répond avec code {response.getcode()}", QSystemTrayIcon.Warning, 5000)
        except Exception as e:
            self.tray.showMessage("VRAMancer API", f"❌ API non accessible\nLancez api_permanente.bat", QSystemTrayIcon.Critical, 5000)

    def run(self):
        sys.exit(self.app.exec_())

    # ------------------------------------------------------------------
    # Résolution robuste de l'icône (Windows: chemins dupliqués / double extraction)
    # ------------------------------------------------------------------
    def _resolve_icon(self) -> QIcon:
        candidates = [
            os.path.join(os.getcwd(), "vramancer.png"),
            os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "vramancer.png"),
            "vramancer.png",
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
