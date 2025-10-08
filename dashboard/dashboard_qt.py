#!/usr/bin/env python3

"""

Dashboard Qt avec monitoring système completfrom PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QListWidget, QListWidgetItem, QHBoxLayout, QTableWidget, QTableWidgetItem

- Affichage ressources PC (CPU, RAM, GPU)from PyQt5.QtGui import QIcon, QPixmap

- Compatible RTX 4060 Laptop GPUfrom PyQt5.QtCore import Qt, QTimer

- Interface native Qtimport os, struct, time

"""try:

	from core.network.network_monitor import NetworkMonitor

import sysexcept Exception:

import os	class NetworkMonitor:  # fallback minimal

import time		def __init__(self): self.stats=[]

try:		def start(self): pass

    import psutiltry:

except ImportError:	from core.network.transmission import send_block

    print("⚠️  psutil non installé - installer avec: pip install psutil")except Exception:

    sys.exit(1)	def send_block(*a, **k): return None

try:

import threading	import torch

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, except Exception:

                           QWidget, QPushButton, QLabel, QTextEdit, QProgressBar,	class _T:

                           QGroupBox, QGridLayout, QTabWidget)		def randn(self,*a,**k): return None

from PyQt5.QtCore import QTimer, QThread, pyqtSignal	torch = _T()

from PyQt5.QtGui import QFonttry:

	import socketio

class SystemMonitor(QThread):except Exception:

    update_signal = pyqtSignal(dict)	socketio = None

    try:

    def run(self):	import requests

        while True:except Exception:

            try:	class _RespStub:

                # Infos système		ok = False

                cpu_percent = psutil.cpu_percent(interval=1)		status_code = 0

                memory = psutil.virtual_memory()		_content = b''

                		def json(self): return {}

                # Disk (cross-platform)		@property

                try:		def content(self): return self._content

                    if os.name == 'nt':  # Windows	class _Req:

                        disk = psutil.disk_usage('C:')		_warned_get = False

                    else:  # Linux/Mac		_warned_post = False

                        disk = psutil.disk_usage('/')		def get(self,*a,**k):

                except:			if not self._warned_get:

                    disk = psutil.disk_usage('.')				print("requests absent (stub) – HTTP GET désactivé (message unique)")

                				self._warned_get = True

                # GPU Info			return _RespStub()

                gpu_info = self.get_gpu_info()		def post(self,*a,**k):

                			if not self._warned_post:

                # Network				print("requests absent (stub) – HTTP POST désactivé (message unique)")

                try:				self._warned_post = True

                    net_io = psutil.net_io_counters()			return _RespStub()

                    net_sent = net_io.bytes_sent // (1024**2)  # MB	requests = _Req()

                    net_recv = net_io.bytes_recv // (1024**2)  # MBtry:

                except:	from core.telemetry import decode_stream

                    net_sent = net_recv = 0except Exception:

                	def decode_stream(blob): return []

                system_data = {

                    'cpu_percent': cpu_percent,

                    'cpu_count': psutil.cpu_count(),class DashboardQt(QWidget):

                    'memory_percent': memory.percent,	def __init__(self):

                    'memory_used': memory.used // (1024**3),  # GB		super().__init__()

                    'memory_total': memory.total // (1024**3),  # GB		self.setWindowTitle("VRAMancer Dashboard Qt")

                    'disk_percent': disk.percent,		self.setGeometry(100, 100, 900, 700)

                    'disk_used': disk.used // (1024**3),  # GB		# ---------------- Configuration réseau / API ----------------

                    'disk_total': disk.total // (1024**3),  # GB		# Auto-détection du port API si VRM_API_BASE non défini: on tente 5030 puis 5010

                    'gpu_info': gpu_info,		self.api_debug = os.environ.get("VRM_API_DEBUG","0") in {"1","true","TRUE"}

                    'net_sent': net_sent,		_env_base = os.environ.get("VRM_API_BASE")

                    'net_recv': net_recv,		if _env_base:

                }			self.api_base = _env_base.rstrip('/')

                		else:

                self.update_signal.emit(system_data)			self.api_base = self._autodetect_api_base()

                time.sleep(2)  # Update every 2 seconds		self.memory_base = os.environ.get("VRM_MEMORY_BASE", "http://localhost:5000").rstrip('/')

                		self.api_timeout = float(os.environ.get("VRM_API_TIMEOUT", "2.5"))

            except Exception as e:		self.api_retries = int(os.environ.get("VRM_API_RETRIES", "3"))

                print(f"Monitor error: {e}")		self._backend_state = None

                time.sleep(5)		layout = QVBoxLayout()

    

    def get_gpu_info(self):		# Supervision des nœuds (API supervision)

        """Récupère les infos GPU (PyTorch + système)"""		self.sup_label = QLabel("Supervision des nœuds:")

        gpu_data = []		layout.addWidget(self.sup_label)

        		self.node_list = QListWidget()

        # PyTorch CUDA		layout.addWidget(self.node_list)

        try:		self.status_label = QLabel("")

            import torch		# Barre statut avec pastille + bouton reconnect

            if torch.cuda.is_available():		from PyQt5.QtWidgets import QHBoxLayout

                for i in range(torch.cuda.device_count()):		status_bar = QHBoxLayout()

                    props = torch.cuda.get_device_properties(i)		self.status_indicator = QLabel("●")

                    memory_used = torch.cuda.memory_allocated(i) // (1024**3)		self.status_indicator.setStyleSheet("font-size:16px;color:gray;margin-right:6px;")

                    memory_total = props.total_memory // (1024**3)		self.reconnect_btn = QPushButton("Reconnect / Refresh")

                    gpu_data.append({		self.reconnect_btn.clicked.connect(self.force_reconnect)

                        'name': props.name,		status_bar.addWidget(self.status_indicator)

                        'memory_used': memory_used,		status_bar.addWidget(self.status_label, 1)

                        'memory_total': memory_total,		status_bar.addWidget(self.reconnect_btn)

                        'memory_percent': (memory_used / memory_total * 100) if memory_total > 0 else 0,		layout.addLayout(status_bar)

                        'backend': 'CUDA',

                        'compute': f"{props.major}.{props.minor}"		self.net_label = QLabel("Network stats:")

                    })		layout.addWidget(self.net_label)

        except Exception as e:		self.net_stats = QTextEdit()

            pass		self.net_stats.setReadOnly(True)

        		layout.addWidget(self.net_stats)

        # VRAMancer devices

        try:		# Vue mémoire hiérarchique

            sys.path.insert(0, os.getcwd())		self.mem_label = QLabel("Mémoire (tiers):")

            from core.utils import enumerate_devices		layout.addWidget(self.mem_label)

            devices = enumerate_devices()		self.mem_table = QTableWidget(0, 5)

            for device in devices:		self.mem_table.setHorizontalHeaderLabels(["ID","Tier","SizeMB","Access","Promote/Demote"])

                if device['backend'] != 'cpu':		layout.addWidget(self.mem_table)

                    gpu_data.append({		self.mem_timer = QTimer(); self.mem_timer.timeout.connect(self.refresh_memory); self.mem_timer.start(4000)

                        'name': device['name'],

                        'backend': device['backend'].upper(),		self.edge_label = QLabel("Edge / IoT Charge Nodes:")

                        'id': device['id']		layout.addWidget(self.edge_label)

                    })		self.edge_stats = QTextEdit(); self.edge_stats.setReadOnly(True)

        except Exception as e:		layout.addWidget(self.edge_stats)

            pass		self.edge_timer = QTimer(); self.edge_timer.timeout.connect(self.refresh_edge); self.edge_timer.start(5000)

        

        return gpu_data		self.offload_btn = QPushButton("Déporter bloc VRAM via USB4")

		self.offload_btn.clicked.connect(self.offload_vram)

class VRAMancerDashboardQt(QMainWindow):		layout.addWidget(self.offload_btn)

    def __init__(self):

        super().__init__()		self.setLayout(layout)

        self.setWindowTitle("🚀 VRAMancer Dashboard Qt - Monitoring RTX 4060 Laptop GPU")

        self.setGeometry(100, 100, 1200, 800)		self.monitor = NetworkMonitor()

        		self.monitor.start()

        # Widget central		self.timer = QTimer()

        central_widget = QWidget()		self.timer.timeout.connect(self.update_stats)

        self.setCentralWidget(central_widget)		self.timer.start(2000)

        

        # Tabs		self.node_timer = QTimer()

        tabs = QTabWidget()		self.node_timer.timeout.connect(self.refresh_nodes)

        central_widget_layout = QVBoxLayout(central_widget)		self.node_timer.start(8000)

        central_widget_layout.addWidget(tabs)

        		# Télémétrie binaire périodique

        # Tab 1: Système		self.telemetry_timer = QTimer(); self.telemetry_timer.timeout.connect(self.fetch_binary_telemetry); self.telemetry_timer.start(4000)

        system_tab = QWidget()

        tabs.addTab(system_tab, "💻 Système")		# SocketIO pour supervision temps réel

        self.setup_system_tab(system_tab)		self.sio = None

        		if socketio:

        # Tab 2: GPU			self._init_socketio()

        gpu_tab = QWidget()		self.nodes = []

        tabs.addTab(gpu_tab, "🎮 GPU RTX 4060")		self.refresh_nodes()

        self.setup_gpu_tab(gpu_tab)

        	def on_nodes(self, data):

        # Tab 3: VRAMancer		self.nodes = data

        vram_tab = QWidget()		self.update_node_list()

        tabs.addTab(vram_tab, "🚀 VRAMancer")

        self.setup_vram_tab(vram_tab)	def on_pong(self, data):

        		node_id = data.get("node_id")

        # Monitor thread		status = data.get("status")

        self.monitor = SystemMonitor()		self.status_label.setText(f"Ping {node_id}: {status}")

        self.monitor.update_signal.connect(self.update_display)

        self.monitor.start()	def refresh_nodes(self):

        		data = self._api_get_json('/api/nodes')

        # Style sombre		if data is not None:

        self.setStyleSheet("""			# Extraire la liste des nodes depuis la réponse API

            QMainWindow { 			if isinstance(data, dict) and 'nodes' in data:

                background-color: #1e1e1e; 				self.nodes = data['nodes']

                color: #ffffff; 			else:

                font-family: 'Segoe UI', Arial, sans-serif;				self.nodes = data

            }			self.update_node_list()

            QTabWidget::pane { 			self._backend_ok()

                border: 2px solid #444; 		else:

                background-color: #1e1e1e; 			self.node_list.clear()

                border-radius: 8px;			self._backend_fail()

            }

            QTabBar::tab { 	def fetch_binary_telemetry(self):

                background-color: #444; 		blob = self._api_get_content('/api/telemetry.bin')

                color: #fff; 		if not blob:

                padding: 12px 20px; 			return

                margin-right: 4px; 		try:

                border-top-left-radius: 8px;			decoded = list(decode_stream(blob))

                border-top-right-radius: 8px;			self.edge_stats.clear()

            }			for n in decoded:

            QTabBar::tab:selected { 				cid = n.get('id','?')

                background-color: #666; 				cpu = n.get('cpu_load_pct',0.0)

                border-bottom: 3px solid #4CAF50;				freec = n.get('free_cores','?')

            }				vused = n.get('vram_used_mb','?'); vtot = n.get('vram_total_mb','?')

            QGroupBox { 				try:

                font-weight: bold; 					cpu_fmt = f"{float(cpu):.2f}"

                border: 2px solid #555; 				except Exception:

                margin: 15px; 					cpu_fmt = str(cpu)

                padding-top: 15px; 				self.edge_stats.append(f"{cid} | load={cpu_fmt}% free={freec} vram={vused}/{vtot}MB")

                border-radius: 10px;		except Exception:

                background-color: #2a2a2a;			pass

            }

            QGroupBox::title { 	def update_node_list(self):

                subcontrol-origin: margin; 		self.node_list.clear()

                left: 15px; 		for node in self.nodes:

                padding: 0 8px;			typ = node.get("type", "standard")

                color: #4CAF50;			icon_path = os.path.join(os.path.dirname(__file__), "../static/icons/", node.get("icon", "standard.svg"))

                font-size: 14px;			if os.path.exists(icon_path):

            }				icon = QIcon(QPixmap(icon_path).scaled(32, 32))

            QProgressBar { 			else:

                border: 2px solid #555; 				icon = QIcon()

                border-radius: 8px; 			item = QListWidgetItem(f"{node['id']} [{typ}] - {node.get('status', 'inconnu')} | CPU:{node.get('cpu','?')} RAM:{node.get('ram','?')} GPU:{node.get('gpu','?')} OS:{node.get('os','?')} Conn:{node.get('conn','?')}")

                text-align: center;			item.setIcon(icon)

                background-color: #333;			self.node_list.addItem(item)

                color: white;		# Ajout d’un bouton d’action par nœud (ping)

                font-weight: bold;		for i in range(self.node_list.count()):

            }			btn = QPushButton("Ping")

            QProgressBar::chunk { 			btn.clicked.connect(lambda _, idx=i: self.send_action(idx))

                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4CAF50, stop:1 #45a049);			self.node_list.setItemWidget(self.node_list.item(i), btn)

                border-radius: 6px;

            }	def send_action(self, idx):

            QLabel { 		if idx < len(self.nodes):

                color: #ffffff; 			node = self.nodes[idx]

                font-size: 12px;			try:

            }				resp = requests.post(f"{self.api_base}/api/nodes/{node['id']}/action", json={"action": "ping"})

            QPushButton { 				if resp.ok:

                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4CAF50, stop:1 #45a049);					self.status_label.setText(f"Action envoyée à {node['id']}")

                color: white; 					if self.sio:

                border: none; 						self.sio.emit("ping", {"node_id": node['id']})

                padding: 10px 15px; 			except Exception as e:

                border-radius: 6px;				self.status_label.setText(f"Erreur action: {e}")

                font-weight: bold;

            }	def update_stats(self):

            QPushButton:hover { 		if self.monitor.stats:

                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #45a049, stop:1 #3d8b40);			last = self.monitor.stats[-1]

            }			self.net_stats.append(f"Sent: {last['sent']/1024:.1f} KB | Recv: {last['recv']/1024:.1f} KB")

            QPushButton:pressed {

                background: #3d8b40;	def offload_vram(self):

            }		try:

            QTextEdit {			if hasattr(torch,'randn'):

                background-color: #2a2a2a;				tensor = torch.randn(512, 512)

                border: 2px solid #555;				send_block([tensor], [getattr(tensor,'shape',())], [str(getattr(tensor,'dtype','f32'))], target_device="machineB", usb4_path="/mnt/usb4_share", protocol="usb4", compress=True)

                border-radius: 8px;				self.net_stats.append("Bloc VRAM transféré via USB4 !")

                color: white;		except Exception:

                font-family: 'Consolas', monospace;			self.net_stats.append("Offload indisponible (deps manquantes)")

                font-size: 11px;

            }	def refresh_memory(self):

        """)		try:

    			resp = requests.get("http://localhost:5000/api/memory")

    def setup_system_tab(self, tab):			if resp.ok:

        layout = QVBoxLayout(tab)				data = resp.json()

        				blocks = data.get("blocks", {})

        # Header info				self.mem_table.setRowCount(len(blocks))

        header = QLabel("💻 Ressources Système en Temps Réel")				for r,(bid, meta) in enumerate(blocks.items()):

        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50; margin: 10px;")					self.mem_table.setItem(r,0,QTableWidgetItem(bid[:8]))

        layout.addWidget(header)					self.mem_table.setItem(r,1,QTableWidgetItem(meta.get("tier","?")))

        					self.mem_table.setItem(r,2,QTableWidgetItem(str(meta.get("size_mb","?"))))

        # CPU Group					self.mem_table.setItem(r,3,QTableWidgetItem(str(meta.get("access","0"))))

        cpu_group = QGroupBox("🔧 Processeur Intel i5 12e gen")					btn_widget = QWidget(); hb = QHBoxLayout(); hb.setContentsMargins(0,0,0,0)

        cpu_layout = QGridLayout(cpu_group)					btn_p = QPushButton("+"); btn_d = QPushButton("-")

        					short = bid[:8]

        self.cpu_label = QLabel("CPU: Initialisation...")					btn_p.clicked.connect(lambda _, b=short: self.promote_block(b))

        self.cpu_progress = QProgressBar()					btn_d.clicked.connect(lambda _, b=short: self.demote_block(b))

        self.cpu_progress.setMaximum(100)					hb.addWidget(btn_p); hb.addWidget(btn_d); btn_widget.setLayout(hb)

        self.cpu_progress.setMinimumHeight(25)					self.mem_table.setCellWidget(r,4, btn_widget)

        		except Exception:

        cpu_layout.addWidget(self.cpu_label, 0, 0)			pass

        cpu_layout.addWidget(self.cpu_progress, 0, 1)

        	def promote_block(self, short_id):

        # Memory Group		self._memory_simple(f"/api/memory/promote?id={short_id}")

        memory_group = QGroupBox("💾 Mémoire RAM")

        memory_layout = QGridLayout(memory_group)	def demote_block(self, short_id):

        		self._memory_simple(f"/api/memory/demote?id={short_id}")

        self.memory_label = QLabel("RAM: Initialisation...")

        self.memory_progress = QProgressBar()	def refresh_edge(self):

        self.memory_progress.setMaximum(100)		data = self._api_get_json('/api/nodes')

        self.memory_progress.setMinimumHeight(25)		if not data:

        			self._backend_fail(); return

        memory_layout.addWidget(self.memory_label, 0, 0)		self.edge_stats.clear()

        memory_layout.addWidget(self.memory_progress, 0, 1)		# Extract nodes array from API response

        		nodes = data['nodes'] if isinstance(data, dict) and 'nodes' in data else data if isinstance(data, list) else []

        # Disk Group		for n in nodes:

        disk_group = QGroupBox("💿 Stockage SSD/HDD")			self.edge_stats.append(f"{n['id']} | type={n.get('type')} | load={n.get('cpu_load_pct','?')}% | free_cores={n.get('free_cores','?')}")

        disk_layout = QGridLayout(disk_group)

        	# -------------------- Helpers HTTP avec retries --------------------

        self.disk_label = QLabel("Disque: Initialisation...")	def _api_get_json(self, path: str):

        self.disk_progress = QProgressBar()		return self._do_get(path, json_mode=True, bases=[self.api_base])

        self.disk_progress.setMaximum(100)

        self.disk_progress.setMinimumHeight(25)	def _api_get_content(self, path: str):

        		return self._do_get(path, json_mode=False, bases=[self.api_base])

        disk_layout.addWidget(self.disk_label, 0, 0)

        disk_layout.addWidget(self.disk_progress, 0, 1)	def _memory_simple(self, path: str):

        		self._do_get(path, json_mode=False, bases=[self.memory_base], silent=True)

        # Network Group

        network_group = QGroupBox("🌐 Réseau")	def _do_get(self, path: str, json_mode=True, bases=None, silent=False):

        network_layout = QVBoxLayout(network_group)		if not requests:

        			return None

        self.network_label = QLabel("Réseau: Initialisation...")		bases = bases or [self.api_base]

        network_layout.addWidget(self.network_label)		# Ajoute fallback 127.0.0.1 si localhost

        		expanded = []

        layout.addWidget(cpu_group)		for b in bases:

        layout.addWidget(memory_group)			expanded.append(b)

        layout.addWidget(disk_group)			if 'localhost' in b:

        layout.addWidget(network_group)				expanded.append(b.replace('localhost','127.0.0.1'))

        layout.addStretch()		for base in expanded:

    			for attempt in range(self.api_retries):

    def setup_gpu_tab(self, tab):				try:

        layout = QVBoxLayout(tab)					if self.api_debug:

        						print(f"[QT][HTTP] GET {base+path} attempt={attempt+1}/{self.api_retries}")

        # Header					resp = requests.get(base + path, timeout=self.api_timeout)

        header = QLabel("🎮 Cartes Graphiques - RTX 4060 Laptop GPU")					if resp.ok:

        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50; margin: 10px;")						if self.api_debug:

        layout.addWidget(header)							print(f"[QT][HTTP] OK {base+path} status={resp.status_code}")

        						if json_mode:

        self.gpu_group = QGroupBox("🎮 GPUs Détectés")							return resp.json()

        self.gpu_layout = QVBoxLayout(self.gpu_group)						return resp.content

        					time.sleep( min(0.6, 0.25 * (attempt+1)) )

        self.gpu_info_label = QLabel("🔄 Détection GPU en cours...")				except Exception as e:

        self.gpu_layout.addWidget(self.gpu_info_label)					if self.api_debug:

        						print(f"[QT][HTTP] FAIL {base+path} attempt={attempt+1} err={e}")

        layout.addWidget(self.gpu_group)					time.sleep( min(0.6, 0.25 * (attempt+1)) )

        layout.addStretch()		if not silent and json_mode:

    			return None

    def setup_vram_tab(self, tab):		return None

        layout = QVBoxLayout(tab)

        	def _autodetect_api_base(self):

        # Header		candidates = [

        header = QLabel("🚀 VRAMancer - Gestionnaire VRAM Intelligent")			f"http://localhost:{os.environ.get('VRM_API_PORT','5030')}",

        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50; margin: 10px;")			"http://localhost:5030",

        layout.addWidget(header)			"http://localhost:5010",

        		]

        # VRAMancer Status		if requests:

        status_group = QGroupBox("📊 Status VRAMancer")			for base in candidates:

        status_layout = QVBoxLayout(status_group)				try:

        					if self.api_debug:

        self.vram_status_label = QLabel("Status: 🔄 Initialisation...")						print(f"[QT][AUTODETECT] probing {base}/api/health")

        self.vram_backend_label = QLabel("Backend: 🔄 Détection...")					r = requests.get(base + '/api/health', timeout=0.8)

        status_layout.addWidget(self.vram_status_label)					if r.ok:

        status_layout.addWidget(self.vram_backend_label)						if self.api_debug:

        							print(f"[QT][AUTODETECT] SELECT {base}")

        # Controls						return base

        controls_group = QGroupBox("🎛️ Contrôles VRAMancer")				except Exception as e:

        controls_layout = QHBoxLayout(controls_group)					if self.api_debug:

        						print(f"[QT][AUTODETECT] fail {base} err={e}")

        self.test_api_btn = QPushButton("🧪 Test API (Port 5030)")					continue

        self.test_api_btn.clicked.connect(self.test_api)		return candidates[0]

        

        self.open_web_btn = QPushButton("🌐 Dashboard Web")	def _backend_ok(self):

        self.open_web_btn.clicked.connect(self.open_web_dashboard)		if self._backend_state != 'ok':

        			self.status_label.setText(f"Connecté {self.api_base}")

        self.open_mobile_btn = QPushButton("📱 Dashboard Mobile")			self._set_indicator('ok')

        self.open_mobile_btn.clicked.connect(self.open_mobile_dashboard)			self._backend_state = 'ok'

        

        controls_layout.addWidget(self.test_api_btn)	def _backend_fail(self):

        controls_layout.addWidget(self.open_web_btn)		if self._backend_state != 'fail':

        controls_layout.addWidget(self.open_mobile_btn)			self.status_label.setText(f"Backend injoignable ({self.api_base}) – lancer: python -m core.api.unified_api (ou définir VRM_API_BASE)")

        			self._set_indicator('fail')

        # Logs			self._backend_state = 'fail'

        logs_group = QGroupBox("📝 Logs VRAMancer")

        logs_layout = QVBoxLayout(logs_group)	# -------------------- UI helpers statut --------------------

        	def _set_indicator(self, state: str):

        self.logs_text = QTextEdit()		if state == 'ok':

        self.logs_text.setMaximumHeight(150)			self.status_indicator.setStyleSheet("font-size:16px;color:#21c55d;margin-right:6px;")  # vert

        self.logs_text.append("=== VRAMancer Dashboard Qt ===")		elif state == 'fail':

        self.logs_text.append("✅ Interface initialisée")			self.status_indicator.setStyleSheet("font-size:16px;color:#dc2626;margin-right:6px;")  # rouge

        logs_layout.addWidget(self.logs_text)		else:

        			self.status_indicator.setStyleSheet("font-size:16px;color:gray;margin-right:6px;")

        layout.addWidget(status_group)

        layout.addWidget(controls_group)	def force_reconnect(self):

        layout.addWidget(logs_group)		# Force une vérification immédiate + tentative socketio

        layout.addStretch()		self.status_label.setText("Reconnexion...")

    		self._set_indicator('pending')

    def update_display(self, data):		self.refresh_nodes()

        # Update CPU		if socketio and (not getattr(self,'sio',None) or not self._socketio_connected()):

        cpu_text = f"CPU: {data['cpu_percent']:.1f}% ({data['cpu_count']} coeurs)"			self._init_socketio(force=True)

        self.cpu_label.setText(cpu_text)

        self.cpu_progress.setValue(int(data['cpu_percent']))	def _socketio_connected(self):

        		try:

        # Color coding for CPU			return bool(self.sio) and self.sio.connected

        if data['cpu_percent'] > 80:		except Exception:

            self.cpu_progress.setStyleSheet("QProgressBar::chunk { background-color: #f44336; }")			return False

        elif data['cpu_percent'] > 60:

            self.cpu_progress.setStyleSheet("QProgressBar::chunk { background-color: #ff9800; }")	def _init_socketio(self, force=False):

        else:		if not socketio:

            self.cpu_progress.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")			return

        		if getattr(self,'sio',None) and self._socketio_connected() and not force:

        # Update Memory			return

        memory_text = f"RAM: {data['memory_used']} / {data['memory_total']} GB ({data['memory_percent']:.1f}%)"		try:

        self.memory_label.setText(memory_text)			self.sio = socketio.Client(reconnection=True, reconnection_attempts=2, reconnection_delay=1)

        self.memory_progress.setValue(int(data['memory_percent']))			self.sio.on("nodes", self.on_nodes)

        			self.sio.on("pong", self.on_pong)

        # Update Disk			self.sio.connect(self.api_base, wait_timeout=2)

        disk_text = f"Disque: {data['disk_used']} / {data['disk_total']} GB ({data['disk_percent']:.1f}%)"		except Exception:

        self.disk_label.setText(disk_text)			self.sio = None

        self.disk_progress.setValue(int(data['disk_percent']))

        if __name__ == "__main__":

        # Update Network	app = QApplication([])

        network_text = f"📤 Envoyé: {data['net_sent']} MB | 📥 Reçu: {data['net_recv']} MB"	win = DashboardQt()

        self.network_label.setText(network_text)	win.show()

        	app.exec_()

        # Update GPU
        self.update_gpu_display(data['gpu_info'])
        
        # Update VRAMancer status
        self.update_vramancer_status()
    
    def update_gpu_display(self, gpu_info):
        # Clear existing GPU info
        for i in reversed(range(self.gpu_layout.count())):
            widget = self.gpu_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        if not gpu_info:
            no_gpu_label = QLabel("❌ Aucun GPU CUDA détecté\n💡 Vérifiez l'installation PyTorch CUDA")
            no_gpu_label.setStyleSheet("color: #ff9800; font-size: 14px;")
            self.gpu_layout.addWidget(no_gpu_label)
        else:
            for i, gpu in enumerate(gpu_info):
                gpu_widget = QWidget()
                gpu_layout = QVBoxLayout(gpu_widget)  # Vertical layout for more info
                
                # GPU Name and backend
                name_text = f"🎮 {gpu['name']} ({gpu['backend']})"
                if 'compute' in gpu:
                    name_text += f" - Compute {gpu['compute']}"
                
                gpu_name_label = QLabel(name_text)
                gpu_name_label.setStyleSheet("font-weight: bold; color: #4CAF50; font-size: 13px;")
                gpu_layout.addWidget(gpu_name_label)
                
                # Memory bar if available
                if 'memory_total' in gpu and gpu['memory_total'] > 0:
                    memory_text = f"VRAM: {gpu['memory_used']}/{gpu['memory_total']} GB ({gpu['memory_percent']:.1f}%)"
                    memory_label = QLabel(memory_text)
                    
                    gpu_progress = QProgressBar()
                    gpu_progress.setMaximum(100)
                    gpu_progress.setValue(int(gpu['memory_percent']))
                    gpu_progress.setMinimumHeight(20)
                    
                    gpu_layout.addWidget(memory_label)
                    gpu_layout.addWidget(gpu_progress)
                
                # Add widget to main layout
                self.gpu_layout.addWidget(gpu_widget)
    
    def update_vramancer_status(self):
        try:
            # Test VRAMancer backend
            sys.path.insert(0, os.getcwd())
            from core.utils import detect_backend, enumerate_devices
            
            backend = detect_backend()
            devices = enumerate_devices()
            
            status_text = f"✅ Opérationnel - {len(devices)} device(s) disponible(s)"
            backend_text = f"Backend actif: {backend.upper()}"
            
            self.vram_status_label.setText(status_text)
            self.vram_backend_label.setText(backend_text)
            
            # Log GPU info
            for device in devices:
                if device['backend'] != 'cpu':
                    self.logs_text.append(f"🎮 GPU détecté: {device['name']} ({device['backend']})")
            
        except Exception as e:
            self.vram_status_label.setText(f"❌ Erreur VRAMancer: {str(e)}")
            self.vram_backend_label.setText("Backend: Non disponible")
    
    def test_api(self):
        try:
            import requests
            self.logs_text.append("🔄 Test API VRAMancer...")
            r = requests.get('http://localhost:5030/health', timeout=3)
            if r.status_code == 200:
                data = r.json()
                self.logs_text.append("✅ API VRAMancer OK - Port 5030")
                self.logs_text.append(f"📊 Status: {data}")
            else:
                self.logs_text.append(f"❌ API erreur HTTP: {r.status_code}")
        except Exception as e:
            self.logs_text.append(f"❌ API inaccessible: {str(e)}")
            self.logs_text.append("💡 Lancez d'abord: api_permanente.bat")
    
    def open_web_dashboard(self):
        import webbrowser
        try:
            webbrowser.open('http://localhost:5000')
            self.logs_text.append("🌐 Dashboard Web ouvert - Port 5000")
        except Exception as e:
            self.logs_text.append(f"❌ Erreur ouverture web: {e}")
    
    def open_mobile_dashboard(self):
        import webbrowser
        try:
            webbrowser.open('http://localhost:5003')
            self.logs_text.append("📱 Dashboard Mobile ouvert - Port 5003")
        except Exception as e:
            self.logs_text.append(f"❌ Erreur ouverture mobile: {e}")

def main():
    app = QApplication(sys.argv)
    
    # Style application
    app.setStyle('Fusion')
    
    window = VRAMancerDashboardQt()
    window.show()
    
    print("=" * 60)
    print("  🚀 VRAMANCER DASHBOARD QT")
    print("=" * 60)
    print("✅ Interface Qt lancée")
    print("🎮 Monitoring RTX 4060 Laptop GPU")
    print("📊 Ressources système en temps réel")
    print("⚙️  Intégration VRAMancer complète")
    print("=" * 60)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()