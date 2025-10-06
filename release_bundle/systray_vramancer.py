import sys
import os
import json
import socket
try:
    from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction, QMessageBox
    from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
    from PyQt5.QtCore import QProcess, QTimer
except Exception as e:  # pragma: no cover - environnement sans GUI
    print("[Systray] PyQt5 indisponible :", e)
    QApplication = QSystemTrayIcon = QMenu = QAction = object  # type: ignore
    QIcon = QProcess = QTimer = object  # type: ignore
    QMessageBox = None

class VRAMancerTray:
    def __init__(self):
        print("[VRAMancer Systray] Démarrage du systray...")
        self.app = QApplication(sys.argv)
        # Détection automatique du chemin du bundle, même si dans un sous-dossier du dépôt
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
        self._base_icon = icon
        self.tray = QSystemTrayIcon(icon, self.app)
        self.menu = QMenu()

        # État persistant
        self._state_path = os.path.join(self.bundle_dir, ".vramancer_systray.json")
        self._state = self._load_state()
        self._tracing_enabled = bool(self._state.get("tracing_enabled", False))
        self._last_dashboards = list(self._state.get("last_dashboards", []))
        self._last_api_port = int(self._state.get("last_api_port", os.environ.get("VRM_API_PORT", 5010)))

        self.install_action = QAction("Installation graphique VRAMancer")
        self.install_action.triggered.connect(self.launch_installer)
        self.menu.addAction(self.install_action)

        # Sous-menu Dashboards
        dashboards_menu = self.menu.addMenu("Dashboards / Modes")

        self.recent_menu = dashboards_menu.addMenu("Derniers (raccourcis)")
        self._rebuild_recent_dashboards()

        act_web = QAction("Web (basique)")
        act_web.triggered.connect(lambda: self.launch_dashboard_variant("dashboard_web.py"))
        dashboards_menu.addAction(act_web)

        act_web_adv = QAction("Web avancé")
        act_web_adv.triggered.connect(lambda: self.launch_dashboard_variant("dashboard_web_advanced.py"))
        dashboards_menu.addAction(act_web_adv)

        act_qt = QAction("Qt GUI")
        act_qt.triggered.connect(lambda: self.launch_dashboard_variant("dashboard_qt.py"))
        dashboards_menu.addAction(act_qt)

        act_tk = QAction("Tk GUI")
        act_tk.triggered.connect(lambda: self.launch_dashboard_variant("dashboard_tk.py"))
        dashboards_menu.addAction(act_tk)

        act_cli = QAction("CLI dashboard")
        act_cli.triggered.connect(lambda: self.launch_dashboard_variant("dashboard_cli.py"))
        dashboards_menu.addAction(act_cli)

        act_visualizer = QAction("Visualiseur (visualizer.py)")
        act_visualizer.triggered.connect(lambda: self.launch_dashboard_variant("visualizer.py"))
        dashboards_menu.addAction(act_visualizer)

        # Fast actions
        fast_menu = self.menu.addMenu("Actions rapides")
        act_api = QAction("Lancer API principale")
        act_api.triggered.connect(self.launch_main_api)
        fast_menu.addAction(act_api)

        act_api_lite = QAction("API Lite (test mode)")
        act_api_lite.triggered.connect(self.launch_api_lite)
        fast_menu.addAction(act_api_lite)

        self.act_trace = QAction(self._trace_label())
        self.act_trace.triggered.connect(self.toggle_tracing)
        fast_menu.addAction(self.act_trace)

        act_metrics = QAction("Ouvrir métriques (curl localhost:9108/metrics)")
        act_metrics.triggered.connect(self.show_metrics_hint)
        fast_menu.addAction(act_metrics)

        act_reload = QAction("Redémarrer (reloader simple)")
        act_reload.triggered.connect(self.simple_reload)
        fast_menu.addAction(act_reload)

        act_ha = QAction("Statut HA")
        act_ha.triggered.connect(self.show_ha_status)
        fast_menu.addAction(act_ha)

        # Documentation
        act_docs = QAction("Ouvrir docs locale")
        act_docs.triggered.connect(self.open_docs)
        self.menu.addAction(act_docs)

        self.quit_action = QAction("Quitter")
        self.quit_action.triggered.connect(self.app.quit)
        self.menu.addAction(self.quit_action)

        self.tray.setContextMenu(self.menu)
        self.tray.setToolTip("VRAMancer est lancé")
        self.tray.show()
        print("[VRAMancer Systray] Systray lancé. Vérifiez la barre de tâches.")

        # Monitoring santé dynamique (toutes les 5s)
        if isinstance(QTimer, type):  # PyQt5 disponible
            self._health_status = "unknown"
            self._health_timer = QTimer()
            self._health_timer.setInterval(5000)
            self._health_timer.timeout.connect(self._refresh_health)
            self._health_timer.start()
            # première vérification immédiate
            self._refresh_health()

    def launch_installer(self):
        # Lance l'installateur graphique
        installer_path = os.path.join(self.bundle_dir, "installer_gui.py")
        if not os.path.exists(installer_path):
            print(f"[ERREUR] Installateur non trouvé : {installer_path}")
        QProcess.startDetached(sys.executable, [installer_path])

    def launch_dashboard_variant(self, filename: str):
        path = os.path.join(self.bundle_dir, "dashboard", filename)
        if not os.path.exists(path):
            print(f"[ERREUR] Dashboard variant non trouvé : {path}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Erreur", f"Dashboard variant non trouvé :\n{path}")
            return
        QProcess.startDetached(sys.executable, [path])
        # Mise à jour récents
        if filename not in self._last_dashboards:
            self._last_dashboards.insert(0, filename)
        self._last_dashboards = self._last_dashboards[:5]
        self._persist_state()
        self._rebuild_recent_dashboards()

    def launch_main_api(self):
        # Lance le serveur principal (Flask + mémoire) via le module principal si disponible
        main_mod = os.path.join(self.bundle_dir, "vramancer", "main.py")
        if not os.path.exists(main_mod):
            # fallback: essayer core/main.py ou racine gui.py
            alt = os.path.join(self.bundle_dir, "gui.py")
            candidate = main_mod if os.path.exists(main_mod) else alt
        else:
            candidate = main_mod
        port = self._choose_port()
        env = os.environ.copy()
        env["VRM_API_PORT"] = str(port)
        self._last_api_port = port
        self._persist_state()
        QProcess.startDetached(sys.executable, [candidate], self.bundle_dir, list(f"{k}={v}" for k,v in env.items()))

    def launch_api_lite(self):
        """Lance l'API en mode test allégé (désactive rate limit + test mode)."""
        main_mod = os.path.join(self.bundle_dir, "vramancer", "main.py")
        if not os.path.exists(main_mod):
            print("[WARN] main.py introuvable pour API lite")
            return
        env = os.environ.copy()
        env.setdefault("VRM_DISABLE_RATE_LIMIT", "1")
        env.setdefault("VRM_TEST_MODE", "1")
        if self._tracing_enabled:
            env["VRM_TRACING"] = "1"
        port = self._choose_port()
        env["VRM_API_PORT"] = str(port)
        self._last_api_port = port
        self._persist_state()
        QProcess.startDetached(sys.executable, [main_mod], self.bundle_dir, list(f"{k}={v}" for k,v in env.items()))

    def show_metrics_hint(self):
        if QMessageBox:
            QMessageBox.information(None, "Métriques", "Les métriques Prometheus sont exposées sur http://localhost:9108/metrics\nUtilisez: curl -s http://localhost:9108/metrics | grep vramancer")
        else:
            print("[INFO] Metrics: http://localhost:9108/metrics")

    def simple_reload(self):
        # Stratégie simple: relancer un script de bootstrap si présent
        bootstrap = os.path.join(self.bundle_dir, "scripts", "bootstrap_env.py")
        if os.path.exists(bootstrap):
            QProcess.startDetached(sys.executable, [bootstrap])
        else:
            print("[INFO] Aucun bootstrap_env.py trouvé pour reload.")

    def open_docs(self):
        index_md = os.path.join(self.bundle_dir, "docs", "Index.md")
        if not os.path.exists(index_md):
            print("[INFO] Docs Index.md introuvable")
            return
        import webbrowser
        webbrowser.open_new_tab(f"file://{index_md}")

    def show_ha_status(self):
        """Parse les métriques HA et affiche taille/rotations."""
        import re, requests
        url = f"http://localhost:{os.environ.get('VRM_METRICS_PORT', '9108')}/metrics"
        size = rotations = None
        try:
            resp = requests.get(url, timeout=1.0)
            if resp.ok:
                for line in resp.text.splitlines():
                    if line.startswith('vramancer_ha_journal_size_bytes '):
                        try:
                            size = int(float(line.split()[-1]))
                        except:  # noqa
                            pass
                    elif line.startswith('vramancer_ha_journal_rotations_total '):
                        try:
                            rotations = int(float(line.split()[-1]))
                        except:  # noqa
                            pass
        except Exception as e:
            print("[HA] Erreur récupération métriques:", e)
        msg = f"Journal taille: {size if size is not None else '?'} bytes\nRotations: {rotations if rotations is not None else '?'}"
        if QMessageBox:
            QMessageBox.information(None, "Statut HA", msg)
        else:
            print("[HA STATUS]", msg)

    # --- Santé & icône dynamique ---
    def _refresh_health(self):  # pragma: no cover - interface graphique
        import requests
        port = str(self._last_api_port or os.environ.get("VRM_API_PORT", "5010"))
        url = f"http://localhost:{port}/api/health"
        new_status = "unknown"
        try:
            r = requests.get(url, timeout=0.7)
            if r.ok:
                new_status = "up"
            else:
                new_status = "down"
        except Exception:
            new_status = "down"
        if new_status != getattr(self, "_health_status", None):
            self._health_status = new_status
            if isinstance(QIcon, type):
                self.tray.setIcon(self._make_icon(new_status))
                self.tray.setToolTip(f"VRAMancer ({new_status})")

    def _make_icon(self, status: str) -> 'QIcon':  # pragma: no cover - graphique
        if not isinstance(QPixmap, type):  # PyQt5 absent
            return self._base_icon
        pix = self._base_icon.pixmap(64, 64)
        painter = QPainter(pix)
        color = QColor("gray")
        if status == "up":
            color = QColor(0, 170, 0)
        elif status == "down":
            color = QColor(200, 0, 0)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(color)
        painter.setPen(color)
        painter.drawEllipse(pix.width()-20, pix.height()-20, 14, 14)
        painter.end()
        return QIcon(pix)

    # --- Persistance & utilitaires ---
    def _load_state(self):
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print("[Systray] Échec lecture état:", e)
        return {}

    def _persist_state(self):
        data = {
            "tracing_enabled": self._tracing_enabled,
            "last_dashboards": self._last_dashboards,
            "last_api_port": self._last_api_port,
        }
        try:
            with open(self._state_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print("[Systray] Échec écriture état:", e)

    def _rebuild_recent_dashboards(self):
        if not hasattr(self, 'recent_menu'):
            return
        self.recent_menu.clear()
        if not self._last_dashboards:
            act = QAction("(vide)")
            act.setEnabled(False)
            self.recent_menu.addAction(act)
            return
        for fname in self._last_dashboards:
            act = QAction(fname)
            act.triggered.connect(lambda checked=False, f=fname: self.launch_dashboard_variant(f))
            self.recent_menu.addAction(act)

    def _trace_label(self):
        return "Tracing ON" if self._tracing_enabled else "Activer tracing"

    def toggle_tracing(self):
        self._tracing_enabled = not self._tracing_enabled
        self.act_trace.setText(self._trace_label())
        self._persist_state()

    def _choose_port(self, start=5010, end=5050):
        # Respecte port existant si libre
        preferred = int(os.environ.get("VRM_API_PORT", start))
        if self._port_free(preferred):
            return preferred
        for p in range(start, end + 1):
            if self._port_free(p):
                return p
        return preferred  # fallback

    def _port_free(self, port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                return True
        except OSError:
            return False

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    tray = VRAMancerTray()
    tray.run()
