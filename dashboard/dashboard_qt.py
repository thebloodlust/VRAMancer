

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QListWidget, QListWidgetItem, QHBoxLayout, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer
import os, struct, time
try:
	from core.network.network_monitor import NetworkMonitor
except Exception:
	class NetworkMonitor:  # fallback minimal
		def __init__(self): self.stats=[]
		def start(self): pass
try:
	from core.network.transmission import send_block
except Exception:
	def send_block(*a, **k): return None
try:
	import torch
except Exception:
	class _T:
		def randn(self,*a,**k): return None
	torch = _T()
try:
	import socketio
except Exception:
	socketio = None
try:
	import requests
except Exception:
	class _RespStub:
		ok = False
		status_code = 0
		_content = b''
		def json(self): return {}
		@property
		def content(self): return self._content
	class _Req:
		_warned_get = False
		_warned_post = False
		def get(self,*a,**k):
			if not self._warned_get:
				print("requests absent (stub) – HTTP GET désactivé (message unique)")
				self._warned_get = True
			return _RespStub()
		def post(self,*a,**k):
			if not self._warned_post:
				print("requests absent (stub) – HTTP POST désactivé (message unique)")
				self._warned_post = True
			return _RespStub()
	requests = _Req()
try:
	from core.telemetry import decode_stream
except Exception:
	def decode_stream(blob): return []


class DashboardQt(QWidget):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("VRAMancer Dashboard Qt")
		self.setGeometry(100, 100, 900, 700)
		# ---------------- Configuration réseau / API ----------------
		# Auto-détection du port API si VRM_API_BASE non défini: on tente 5030 puis 5010
		_env_base = os.environ.get("VRM_API_BASE")
		if _env_base:
			self.api_base = _env_base.rstrip('/')
		else:
			self.api_base = self._autodetect_api_base()
		self.memory_base = os.environ.get("VRM_MEMORY_BASE", "http://localhost:5000").rstrip('/')
		self.api_timeout = float(os.environ.get("VRM_API_TIMEOUT", "2.5"))
		self.api_retries = int(os.environ.get("VRM_API_RETRIES", "3"))
		self._backend_state = None
		layout = QVBoxLayout()

		# Supervision des nœuds (API supervision)
		self.sup_label = QLabel("Supervision des nœuds:")
		layout.addWidget(self.sup_label)
		self.node_list = QListWidget()
		layout.addWidget(self.node_list)
		self.status_label = QLabel("")
		# Barre statut avec pastille + bouton reconnect
		from PyQt5.QtWidgets import QHBoxLayout
		status_bar = QHBoxLayout()
		self.status_indicator = QLabel("●")
		self.status_indicator.setStyleSheet("font-size:16px;color:gray;margin-right:6px;")
		self.reconnect_btn = QPushButton("Reconnect / Refresh")
		self.reconnect_btn.clicked.connect(self.force_reconnect)
		status_bar.addWidget(self.status_indicator)
		status_bar.addWidget(self.status_label, 1)
		status_bar.addWidget(self.reconnect_btn)
		layout.addLayout(status_bar)

		self.net_label = QLabel("Network stats:")
		layout.addWidget(self.net_label)
		self.net_stats = QTextEdit()
		self.net_stats.setReadOnly(True)
		layout.addWidget(self.net_stats)

		# Vue mémoire hiérarchique
		self.mem_label = QLabel("Mémoire (tiers):")
		layout.addWidget(self.mem_label)
		self.mem_table = QTableWidget(0, 5)
		self.mem_table.setHorizontalHeaderLabels(["ID","Tier","SizeMB","Access","Promote/Demote"])
		layout.addWidget(self.mem_table)
		self.mem_timer = QTimer(); self.mem_timer.timeout.connect(self.refresh_memory); self.mem_timer.start(4000)

		self.edge_label = QLabel("Edge / IoT Charge Nodes:")
		layout.addWidget(self.edge_label)
		self.edge_stats = QTextEdit(); self.edge_stats.setReadOnly(True)
		layout.addWidget(self.edge_stats)
		self.edge_timer = QTimer(); self.edge_timer.timeout.connect(self.refresh_edge); self.edge_timer.start(5000)

		self.offload_btn = QPushButton("Déporter bloc VRAM via USB4")
		self.offload_btn.clicked.connect(self.offload_vram)
		layout.addWidget(self.offload_btn)

		self.setLayout(layout)

		self.monitor = NetworkMonitor()
		self.monitor.start()
		self.timer = QTimer()
		self.timer.timeout.connect(self.update_stats)
		self.timer.start(2000)

		self.node_timer = QTimer()
		self.node_timer.timeout.connect(self.refresh_nodes)
		self.node_timer.start(8000)

		# Télémétrie binaire périodique
		self.telemetry_timer = QTimer(); self.telemetry_timer.timeout.connect(self.fetch_binary_telemetry); self.telemetry_timer.start(4000)

		# SocketIO pour supervision temps réel
		self.sio = None
		if socketio:
			self._init_socketio()
		self.nodes = []
		self.refresh_nodes()

	def on_nodes(self, data):
		self.nodes = data
		self.update_node_list()

	def on_pong(self, data):
		node_id = data.get("node_id")
		status = data.get("status")
		self.status_label.setText(f"Ping {node_id}: {status}")

	def refresh_nodes(self):
		data = self._api_get_json('/api/nodes')
		if data is not None:
			self.nodes = data
			self.update_node_list()
			self._backend_ok()
		else:
			self.node_list.clear()
			self._backend_fail()

	def fetch_binary_telemetry(self):
		blob = self._api_get_content('/api/telemetry.bin')
		if not blob:
			return
		try:
			decoded = list(decode_stream(blob))
			self.edge_stats.clear()
			for n in decoded:
				cid = n.get('id','?')
				cpu = n.get('cpu_load_pct',0.0)
				freec = n.get('free_cores','?')
				vused = n.get('vram_used_mb','?'); vtot = n.get('vram_total_mb','?')
				try:
					cpu_fmt = f"{float(cpu):.2f}"
				except Exception:
					cpu_fmt = str(cpu)
				self.edge_stats.append(f"{cid} | load={cpu_fmt}% free={freec} vram={vused}/{vtot}MB")
		except Exception:
			pass

	def update_node_list(self):
		self.node_list.clear()
		for node in self.nodes:
			typ = node.get("type", "standard")
			icon_path = os.path.join(os.path.dirname(__file__), "../static/icons/", node.get("icon", "standard.svg"))
			if os.path.exists(icon_path):
				icon = QIcon(QPixmap(icon_path).scaled(32, 32))
			else:
				icon = QIcon()
			item = QListWidgetItem(f"{node['id']} [{typ}] - {node.get('status', 'inconnu')} | CPU:{node.get('cpu','?')} RAM:{node.get('ram','?')} GPU:{node.get('gpu','?')} OS:{node.get('os','?')} Conn:{node.get('conn','?')}")
			item.setIcon(icon)
			self.node_list.addItem(item)
		# Ajout d’un bouton d’action par nœud (ping)
		for i in range(self.node_list.count()):
			btn = QPushButton("Ping")
			btn.clicked.connect(lambda _, idx=i: self.send_action(idx))
			self.node_list.setItemWidget(self.node_list.item(i), btn)

	def send_action(self, idx):
		if idx < len(self.nodes):
			node = self.nodes[idx]
			try:
				resp = requests.post(f"{self.api_base}/api/nodes/{node['id']}/action", json={"action": "ping"})
				if resp.ok:
					self.status_label.setText(f"Action envoyée à {node['id']}")
					if self.sio:
						self.sio.emit("ping", {"node_id": node['id']})
			except Exception as e:
				self.status_label.setText(f"Erreur action: {e}")

	def update_stats(self):
		if self.monitor.stats:
			last = self.monitor.stats[-1]
			self.net_stats.append(f"Sent: {last['sent']/1024:.1f} KB | Recv: {last['recv']/1024:.1f} KB")

	def offload_vram(self):
		try:
			if hasattr(torch,'randn'):
				tensor = torch.randn(512, 512)
				send_block([tensor], [getattr(tensor,'shape',())], [str(getattr(tensor,'dtype','f32'))], target_device="machineB", usb4_path="/mnt/usb4_share", protocol="usb4", compress=True)
				self.net_stats.append("Bloc VRAM transféré via USB4 !")
		except Exception:
			self.net_stats.append("Offload indisponible (deps manquantes)")

	def refresh_memory(self):
		try:
			resp = requests.get("http://localhost:5000/api/memory")
			if resp.ok:
				data = resp.json()
				blocks = data.get("blocks", {})
				self.mem_table.setRowCount(len(blocks))
				for r,(bid, meta) in enumerate(blocks.items()):
					self.mem_table.setItem(r,0,QTableWidgetItem(bid[:8]))
					self.mem_table.setItem(r,1,QTableWidgetItem(meta.get("tier","?")))
					self.mem_table.setItem(r,2,QTableWidgetItem(str(meta.get("size_mb","?"))))
					self.mem_table.setItem(r,3,QTableWidgetItem(str(meta.get("access","0"))))
					btn_widget = QWidget(); hb = QHBoxLayout(); hb.setContentsMargins(0,0,0,0)
					btn_p = QPushButton("+"); btn_d = QPushButton("-")
					short = bid[:8]
					btn_p.clicked.connect(lambda _, b=short: self.promote_block(b))
					btn_d.clicked.connect(lambda _, b=short: self.demote_block(b))
					hb.addWidget(btn_p); hb.addWidget(btn_d); btn_widget.setLayout(hb)
					self.mem_table.setCellWidget(r,4, btn_widget)
		except Exception:
			pass

	def promote_block(self, short_id):
		self._memory_simple(f"/api/memory/promote?id={short_id}")

	def demote_block(self, short_id):
		self._memory_simple(f"/api/memory/demote?id={short_id}")

	def refresh_edge(self):
		data = self._api_get_json('/api/nodes')
		if not data:
			self._backend_fail(); return
		self.edge_stats.clear()
		for n in data:
			self.edge_stats.append(f"{n['id']} | type={n.get('type')} | load={n.get('cpu_load_pct','?')}% | free_cores={n.get('free_cores','?')}")

	# -------------------- Helpers HTTP avec retries --------------------
	def _api_get_json(self, path: str):
		return self._do_get(path, json_mode=True, bases=[self.api_base])

	def _api_get_content(self, path: str):
		return self._do_get(path, json_mode=False, bases=[self.api_base])

	def _memory_simple(self, path: str):
		self._do_get(path, json_mode=False, bases=[self.memory_base], silent=True)

	def _do_get(self, path: str, json_mode=True, bases=None, silent=False):
		if not requests:
			return None
		bases = bases or [self.api_base]
		# Ajoute fallback 127.0.0.1 si localhost
		expanded = []
		for b in bases:
			expanded.append(b)
			if 'localhost' in b:
				expanded.append(b.replace('localhost','127.0.0.1'))
		for base in expanded:
			for attempt in range(self.api_retries):
				try:
					resp = requests.get(base + path, timeout=self.api_timeout)
					if resp.ok:
						if json_mode:
							return resp.json()
						return resp.content
					time.sleep( min(0.6, 0.25 * (attempt+1)) )
				except Exception:
					time.sleep( min(0.6, 0.25 * (attempt+1)) )
		if not silent and json_mode:
			return None
		return None

	def _autodetect_api_base(self):
		candidates = [
			f"http://localhost:{os.environ.get('VRM_API_PORT','5030')}",
			"http://localhost:5030",
			"http://localhost:5010",
		]
		if requests:
			for base in candidates:
				try:
					r = requests.get(base + '/api/health', timeout=0.8)
					if r.ok:
						return base
				except Exception:
					continue
		return candidates[0]

	def _backend_ok(self):
		if self._backend_state != 'ok':
			self.status_label.setText(f"Connecté {self.api_base}")
			self._set_indicator('ok')
			self._backend_state = 'ok'

	def _backend_fail(self):
		if self._backend_state != 'fail':
			self.status_label.setText(f"Backend injoignable ({self.api_base}) – lancer: python -m core.api.unified_api (ou définir VRM_API_BASE)")
			self._set_indicator('fail')
			self._backend_state = 'fail'

	# -------------------- UI helpers statut --------------------
	def _set_indicator(self, state: str):
		if state == 'ok':
			self.status_indicator.setStyleSheet("font-size:16px;color:#21c55d;margin-right:6px;")  # vert
		elif state == 'fail':
			self.status_indicator.setStyleSheet("font-size:16px;color:#dc2626;margin-right:6px;")  # rouge
		else:
			self.status_indicator.setStyleSheet("font-size:16px;color:gray;margin-right:6px;")

	def force_reconnect(self):
		# Force une vérification immédiate + tentative socketio
		self.status_label.setText("Reconnexion...")
		self._set_indicator('pending')
		self.refresh_nodes()
		if socketio and (not getattr(self,'sio',None) or not self._socketio_connected()):
			self._init_socketio(force=True)

	def _socketio_connected(self):
		try:
			return bool(self.sio) and self.sio.connected
		except Exception:
			return False

	def _init_socketio(self, force=False):
		if not socketio:
			return
		if getattr(self,'sio',None) and self._socketio_connected() and not force:
			return
		try:
			self.sio = socketio.Client(reconnection=True, reconnection_attempts=2, reconnection_delay=1)
			self.sio.on("nodes", self.on_nodes)
			self.sio.on("pong", self.on_pong)
			self.sio.connect(self.api_base, wait_timeout=2)
		except Exception:
			self.sio = None

if __name__ == "__main__":
	app = QApplication([])
	win = DashboardQt()
	win.show()
	app.exec_()
