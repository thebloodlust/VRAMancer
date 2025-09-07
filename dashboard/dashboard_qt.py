# dashboard/dashboard_qt.py
from PySide6.QtWidgets import QApplication, QMainWindow
import sys

class QtWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VRAMancer (Qt)")

def launch_dashboard():
    app = QApplication(sys.argv)
    win = QtWindow()
    win.show()
    sys.exit(app.exec())
