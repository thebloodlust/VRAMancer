import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QMessageBox
)
from PyQt5.QtGui import QFont, QIcon, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from core.gpu_interface import get_available_gpus
import os

# 🎨 Mode sombre
def apply_dark_theme(app):
    dark_style = """
        QWidget {
            background-color: #121212;
            color: #FFFFFF;
            font-family: Arial;
        }
        QPushButton {
            background-color: #1E1E1E;
            border: 1px solid #333;
            padding: 6px;
        }
        QTextEdit {
            background-color: #1E1E1E;
            border: 1px solid #333;
        }
    """
    app.setStyleSheet(dark_style)

# 📊 Widget graphique
class VRAMGraph(Canvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def update_graph(self, gpus):
        self.ax.clear()
        names = [gpu["name"] for gpu in gpus]
        vram = [gpu["total_vram_mb"] for gpu in gpus]
        self.ax.bar(names, vram, color="#00BFFF")
        self.ax.set_title("VRAM disponible par GPU")
        self.ax.set_ylabel("VRAM (MB)")
        self.draw()

# 🚀 Interface principale

class VRAMancerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VRAMancer Monitor")
        self.setGeometry(100, 100, 700, 500)

        # Définir l'icône de la fenêtre
        icon_path = os.path.join(os.path.dirname(__file__), "../vramancer.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        layout = QVBoxLayout()

        # Ajouter le fond d'écran (image en haut du dashboard)
        if os.path.exists(icon_path):
            from PyQt5.QtWidgets import QLabel
            label = QLabel()
            pixmap = QPixmap(icon_path)
            label.setPixmap(pixmap.scaledToWidth(300))
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)

        self.button = QPushButton("🔍 Scanner les GPU")
        self.button.setFont(QFont("Arial", 12))
        self.button.clicked.connect(self.show_gpu_info)
        layout.addWidget(self.button)

        self.text = QTextEdit()
        self.text.setFont(QFont("Courier", 10))
        self.text.setReadOnly(True)
        layout.addWidget(self.text)

        self.graph = VRAMGraph(self)
        layout.addWidget(self.graph)

        self.setLayout(layout)

    def show_gpu_info(self):
        gpus = get_available_gpus()
        self.text.clear()

        if not gpus:
            self.text.setText("Aucun GPU détecté.")
            return

        for gpu in gpus:
            status = "✅ Disponible" if gpu["is_available"] else "❌ Indisponible"
            line = f"{status}\nID: {gpu['id']}\nNom: {gpu['name']}\nVRAM: {gpu['total_vram_mb']} MB\n---\n"
            self.text.append(line)

        self.graph.update_graph(gpus)
        self.check_vram_alert(gpus)

    def check_vram_alert(self, gpus):
        for gpu in gpus:
            if gpu["is_available"] and gpu["total_vram_mb"] < 500:
                alert = QMessageBox()
                alert.setWindowTitle("⚠️ Alerte VRAM")
                alert.setText(f"Le GPU {gpu['name']} est presque saturé ({gpu['total_vram_mb']} MB restants) !")
                alert.setIcon(QMessageBox.Warning)
                alert.exec_()

# 🏁 Lancement
if __name__ == "__main__":
    from PyQt5.QtCore import Qt
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = VRAMancerGUI()
    window.show()
    sys.exit(app.exec_())
