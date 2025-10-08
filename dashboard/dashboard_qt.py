#!/usr/bin/env python3
"""
Dashboard Qt avec monitoring système complet
- Affichage ressources PC (CPU, RAM, GPU)  
- Compatible RTX 4060 Laptop GPU
- Interface native Qt
"""

import sys
import os
import time

try:
    import psutil
except ImportError:
    print("⚠️  psutil non installé - installer avec: pip install psutil")
    input("Appuyez sur Entrée pour fermer...")
    sys.exit(1)

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QTextEdit, QProgressBar,
                            QGroupBox, QGridLayout, QTabWidget, QScrollArea)
    from PyQt5.QtCore import QTimer, QThread, pyqtSignal
    from PyQt5.QtGui import QFont
except ImportError:
    print("⚠️  PyQt5 non installé - installer avec: pip install PyQt5")
    input("Appuyez sur Entrée pour fermer...")
    sys.exit(1)

class SystemMonitor(QThread):
    update_signal = pyqtSignal(dict)
    
    def run(self):
        while True:
            try:
                # Infos système
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Disk
                try:
                    if os.name == 'nt':  # Windows
                        disk = psutil.disk_usage('C:')
                    else:
                        disk = psutil.disk_usage('/')
                except:
                    disk = psutil.disk_usage('.')
                
                # GPU Info
                gpu_info = self.get_gpu_info()
                
                # Network
                try:
                    net_io = psutil.net_io_counters()
                    net_sent = net_io.bytes_sent // (1024**2)
                    net_recv = net_io.bytes_recv // (1024**2)
                except:
                    net_sent = net_recv = 0
                
                system_data = {
                    'cpu_percent': cpu_percent,
                    'cpu_count': psutil.cpu_count(),
                    'memory_percent': memory.percent,
                    'memory_used': memory.used // (1024**3),
                    'memory_total': memory.total // (1024**3),
                    'disk_percent': disk.percent,
                    'disk_used': disk.used // (1024**3),
                    'disk_total': disk.total // (1024**3),
                    'gpu_info': gpu_info,
                    'net_sent': net_sent,
                    'net_recv': net_recv,
                }
                
                self.update_signal.emit(system_data)
                time.sleep(2)
                
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(5)
    
    def get_gpu_info(self):
        """Récupère infos GPU"""
        gpu_data = []
        
        # PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_used = torch.cuda.memory_allocated(i) // (1024**3)
                    memory_total = props.total_memory // (1024**3)
                    gpu_data.append({
                        'name': props.name,
                        'memory_used': memory_used,
                        'memory_total': memory_total,
                        'memory_percent': (memory_used / memory_total * 100) if memory_total > 0 else 0,
                        'backend': 'CUDA',
                        'compute': f"{props.major}.{props.minor}"
                    })
        except:
            pass
        
        # VRAMancer devices
        try:
            sys.path.insert(0, os.getcwd())
            from core.utils import enumerate_devices
            devices = enumerate_devices()
            for device in devices:
                if device['backend'] != 'cpu':
                    gpu_data.append({
                        'name': device['name'],
                        'backend': device['backend'].upper(),
                        'id': device['id']
                    })
        except:
            pass
        
        return gpu_data

class VRAMancerDashboardQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 VRAMancer Dashboard Qt - RTX 4060 Laptop GPU")
        self.setGeometry(100, 100, 1100, 750)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header = QLabel("🚀 VRAMancer Dashboard Qt - Monitoring Système RTX 4060")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #4CAF50; margin: 10px; text-align: center;")
        main_layout.addWidget(header)
        
        # Tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Tab 1: Système
        system_tab = QWidget()
        tabs.addTab(system_tab, "💻 Système")
        self.setup_system_tab(system_tab)
        
        # Tab 2: GPU
        gpu_tab = QWidget()
        tabs.addTab(gpu_tab, "🎮 GPU")
        self.setup_gpu_tab(gpu_tab)
        
        # Tab 3: VRAMancer
        vram_tab = QWidget()
        tabs.addTab(vram_tab, "🚀 VRAMancer")
        self.setup_vram_tab(vram_tab)
        
        # Monitor thread
        self.monitor = SystemMonitor()
        self.monitor.update_signal.connect(self.update_display)
        self.monitor.start()
        
        # Style
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QTabWidget::pane { border: 2px solid #444; background-color: #1e1e1e; }
            QTabBar::tab { background-color: #444; color: #fff; padding: 10px 15px; margin-right: 2px; }
            QTabBar::tab:selected { background-color: #666; }
            QGroupBox { font-weight: bold; border: 2px solid #555; margin: 10px; padding-top: 10px; border-radius: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #4CAF50; }
            QProgressBar { border: 2px solid #555; border-radius: 5px; text-align: center; background-color: #333; }
            QProgressBar::chunk { background-color: #4CAF50; }
            QLabel { color: #ffffff; }
            QPushButton { background-color: #4CAF50; color: white; border: none; padding: 8px 12px; border-radius: 4px; }
            QPushButton:hover { background-color: #45a049; }
        """)
    
    def setup_system_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # CPU
        cpu_group = QGroupBox("🔧 Processeur")
        cpu_layout = QGridLayout(cpu_group)
        self.cpu_label = QLabel("CPU: --")
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setMaximum(100)
        cpu_layout.addWidget(self.cpu_label, 0, 0)
        cpu_layout.addWidget(self.cpu_progress, 0, 1)
        
        # Memory
        memory_group = QGroupBox("💾 Mémoire")
        memory_layout = QGridLayout(memory_group)
        self.memory_label = QLabel("RAM: --")
        self.memory_progress = QProgressBar()
        self.memory_progress.setMaximum(100)
        memory_layout.addWidget(self.memory_label, 0, 0)
        memory_layout.addWidget(self.memory_progress, 0, 1)
        
        # Disk
        disk_group = QGroupBox("💿 Stockage")
        disk_layout = QGridLayout(disk_group)
        self.disk_label = QLabel("Disque: --")
        self.disk_progress = QProgressBar()
        self.disk_progress.setMaximum(100)
        disk_layout.addWidget(self.disk_label, 0, 0)
        disk_layout.addWidget(self.disk_progress, 0, 1)
        
        # Network
        network_group = QGroupBox("🌐 Réseau")
        network_layout = QVBoxLayout(network_group)
        self.network_label = QLabel("Réseau: --")
        network_layout.addWidget(self.network_label)
        
        layout.addWidget(cpu_group)
        layout.addWidget(memory_group)
        layout.addWidget(disk_group)
        layout.addWidget(network_group)
        layout.addStretch()
    
    def setup_gpu_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        self.gpu_group = QGroupBox("🎮 GPUs Détectés")
        self.gpu_layout = QVBoxLayout(self.gpu_group)
        
        self.gpu_info_label = QLabel("🔄 Détection GPU...")
        self.gpu_layout.addWidget(self.gpu_info_label)
        
        layout.addWidget(self.gpu_group)
        layout.addStretch()
    
    def setup_vram_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Status
        status_group = QGroupBox("📊 Status VRAMancer")
        status_layout = QVBoxLayout(status_group)
        self.vram_status_label = QLabel("Status: 🔄 Init...")
        status_layout.addWidget(self.vram_status_label)
        
        # Controls
        controls_group = QGroupBox("🎛️ Contrôles")
        controls_layout = QHBoxLayout(controls_group)
        
        self.test_api_btn = QPushButton("🧪 Test API")
        self.test_api_btn.clicked.connect(self.test_api)
        
        self.open_web_btn = QPushButton("🌐 Web")
        self.open_web_btn.clicked.connect(self.open_web)
        
        self.open_mobile_btn = QPushButton("📱 Mobile")
        self.open_mobile_btn.clicked.connect(self.open_mobile)
        
        controls_layout.addWidget(self.test_api_btn)
        controls_layout.addWidget(self.open_web_btn)
        controls_layout.addWidget(self.open_mobile_btn)
        
        # Logs
        logs_group = QGroupBox("📝 Logs")
        logs_layout = QVBoxLayout(logs_group)
        self.logs_text = QTextEdit()
        self.logs_text.setMaximumHeight(120)
        logs_layout.addWidget(self.logs_text)
        
        layout.addWidget(status_group)
        layout.addWidget(controls_group)
        layout.addWidget(logs_group)
        layout.addStretch()
    
    def update_display(self, data):
        # CPU
        self.cpu_label.setText(f"CPU: {data['cpu_percent']:.1f}% ({data['cpu_count']} cores)")
        self.cpu_progress.setValue(int(data['cpu_percent']))
        
        # Memory
        self.memory_label.setText(f"RAM: {data['memory_used']}/{data['memory_total']} GB ({data['memory_percent']:.1f}%)")
        self.memory_progress.setValue(int(data['memory_percent']))
        
        # Disk
        self.disk_label.setText(f"Disque: {data['disk_used']}/{data['disk_total']} GB ({data['disk_percent']:.1f}%)")
        self.disk_progress.setValue(int(data['disk_percent']))
        
        # Network
        self.network_label.setText(f"📤 {data['net_sent']} MB | 📥 {data['net_recv']} MB")
        
        # GPU
        self.update_gpu_display(data['gpu_info'])
        self.update_vramancer_status()
    
    def update_gpu_display(self, gpu_info):
        # Clear
        for i in reversed(range(self.gpu_layout.count())):
            widget = self.gpu_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        if not gpu_info:
            no_gpu = QLabel("❌ Aucun GPU CUDA détecté")
            self.gpu_layout.addWidget(no_gpu)
        else:
            for gpu in gpu_info:
                gpu_widget = QWidget()
                gpu_layout = QVBoxLayout(gpu_widget)
                
                name_label = QLabel(f"🎮 {gpu['name']} ({gpu['backend']})")
                name_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
                gpu_layout.addWidget(name_label)
                
                if 'memory_total' in gpu and gpu['memory_total'] > 0:
                    mem_text = f"VRAM: {gpu['memory_used']}/{gpu['memory_total']} GB"
                    mem_label = QLabel(mem_text)
                    gpu_layout.addWidget(mem_label)
                    
                    mem_progress = QProgressBar()
                    mem_progress.setMaximum(100)
                    mem_progress.setValue(int(gpu['memory_percent']))
                    gpu_layout.addWidget(mem_progress)
                
                self.gpu_layout.addWidget(gpu_widget)
    
    def update_vramancer_status(self):
        try:
            sys.path.insert(0, os.getcwd())
            from core.utils import detect_backend, enumerate_devices
            
            backend = detect_backend()
            devices = enumerate_devices()
            
            self.vram_status_label.setText(f"✅ Backend: {backend} | Devices: {len(devices)}")
            
        except Exception as e:
            self.vram_status_label.setText(f"❌ Erreur: {str(e)}")
    
    def test_api(self):
        try:
            import requests
            r = requests.get('http://localhost:5030/health', timeout=2)
            if r.status_code == 200:
                self.logs_text.append("✅ API OK")
            else:
                self.logs_text.append(f"❌ API erreur: {r.status_code}")
        except Exception as e:
            self.logs_text.append(f"❌ API: {str(e)}")
    
    def open_web(self):
        import webbrowser
        webbrowser.open('http://localhost:5000')
        self.logs_text.append("🌐 Web ouvert")
    
    def open_mobile(self):
        import webbrowser
        webbrowser.open('http://localhost:5003')
        self.logs_text.append("📱 Mobile ouvert")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = VRAMancerDashboardQt()
    window.show()
    
    print("=" * 50)
    print("🚀 VRAMancer Dashboard Qt")
    print("✅ Interface lancée")
    print("🎮 Monitoring RTX 4060 Laptop GPU")
    print("=" * 50)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()