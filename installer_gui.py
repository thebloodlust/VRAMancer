import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QComboBox, QCheckBox, QMessageBox
)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt
import subprocess

class InstallerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VRAMancer Installer")
        self.setGeometry(200, 200, 600, 400)
        self.setWindowIcon(QIcon("vramancer.png"))
        layout = QVBoxLayout()

        self.mode_label = QLabel("Choose installation mode:")
        self.mode_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Beginner (guided)", "Expert (custom)"])
        layout.addWidget(self.mode_combo)

        import platform
        if platform.system() != "Windows":
            self.option_deb = QCheckBox("Install .deb package (recommended)")
            self.option_deb.setChecked(True)
            layout.addWidget(self.option_deb)

        self.option_lite = QCheckBox("Install Lite CLI version")
        layout.addWidget(self.option_lite)

        self.install_btn = QPushButton("Start Installation")
        self.install_btn.setFont(QFont("Arial", 12))
        self.install_btn.clicked.connect(self.run_install)
        layout.addWidget(self.install_btn)

        self.log_box = QTextEdit()
        self.log_box.setFont(QFont("Courier", 10))
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        self.setLayout(layout)

    def log(self, msg, color="#00FF00"):
        self.log_box.append(f'<span style="color:{color}">{msg}</span>')

    def run_install(self):
        mode = self.mode_combo.currentText()
        self.log(f"Mode selected: {mode}", "#FFD700")
        if self.option_deb.isChecked():
            self.log("Building .deb package...", "#00BFFF")
            try:
                result = subprocess.run(["make", "deb"], capture_output=True, text=True)
                self.log(result.stdout, "#00FF00")
                if result.returncode != 0:
                    self.log(result.stderr, "#FF0000")
                    QMessageBox.critical(self, "Error", ".deb build failed!")
                    return
            except Exception as e:
                self.log(str(e), "#FF0000")
                QMessageBox.critical(self, "Error", str(e))
                return
        if self.option_lite.isChecked():
            self.log("Building Lite CLI version...", "#00BFFF")
            try:
                result = subprocess.run(["make", "lite"], capture_output=True, text=True)
                self.log(result.stdout, "#00FF00")
                if result.returncode != 0:
                    self.log(result.stderr, "#FF0000")
                    QMessageBox.critical(self, "Error", "Lite build failed!")
                    return
            except Exception as e:
                self.log(str(e), "#FF0000")
                QMessageBox.critical(self, "Error", str(e))
                return
        self.log("Installation complete!", "#00FF00")
        QMessageBox.information(self, "Success", "VRAMancer installation finished!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InstallerGUI()
    window.show()
    sys.exit(app.exec_())
