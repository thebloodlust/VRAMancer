

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QListWidget, QListWidgetItem, QHBoxLayout
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer
from core.network.network_monitor import NetworkMonitor
from core.network.transmission import send_block
from core.network.supervision import NodeSupervisor
from core.network.edge_iot import EdgeNode
import torch
import requests
import os
import socketio


class DashboardQt(QWidget):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("VRAMancer Dashboard Qt")
		self.setGeometry(100, 100, 900, 700)
		layout = QVBoxLayout()

		# Supervision des nœuds (API supervision)
		self.sup_label = QLabel("Supervision des nœuds:")
		layout.addWidget(self.sup_label)
		self.node_list = QListWidget()
		layout.addWidget(self.node_list)
		self.status_label = QLabel("")
		layout.addWidget(self.status_label)

		self.net_label = QLabel("Network stats:")
		layout.addWidget(self.net_label)
		self.net_stats = QTextEdit()
		self.net_stats.setReadOnly(True)
		layout.addWidget(self.net_stats)

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
		self.node_timer.start(5000)

		# SocketIO pour supervision temps réel
		self.sio = socketio.Client()
		self.sio.on("nodes", self.on_nodes)
		self.sio.on("pong", self.on_pong)
		try:
			self.sio.connect("http://localhost:5010")
		except Exception:
			pass
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
		try:
			resp = requests.get("http://localhost:5010/api/nodes")
			self.nodes = resp.json()
			self.update_node_list()
		except Exception as e:
			self.node_list.clear()
			self.status_label.setText(f"Erreur: {e}")

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
				resp = requests.post(f"http://localhost:5010/api/nodes/{node['id']}/action", json={"action": "ping"})
				if resp.ok:
					self.status_label.setText(f"Action envoyée à {node['id']}")
					self.sio.emit("ping", {"node_id": node['id']})
			except Exception as e:
				self.status_label.setText(f"Erreur action: {e}")

	def update_stats(self):
		if self.monitor.stats:
			last = self.monitor.stats[-1]
			self.net_stats.append(f"Sent: {last['sent']/1024:.1f} KB | Recv: {last['recv']/1024:.1f} KB")

	def offload_vram(self):
		tensor = torch.randn(512, 512)
		send_block([tensor], [tensor.shape], [str(tensor.dtype)], target_device="machineB", usb4_path="/mnt/usb4_share", protocol="usb4", compress=True)
		self.net_stats.append("Bloc VRAM transféré via USB4 !")

if __name__ == "__main__":
	app = QApplication([])
	win = DashboardQt()
	win.show()
	app.exec_()
