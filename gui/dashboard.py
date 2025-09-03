import tkinter as tk
from tkinter import ttk
from core.monitor import GPUMonitor
from core.stream_manager import StreamManager
from core.logger import Logger
from core.scheduler import Scheduler

class DashboardGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VRAMancer Dashboard Deluxe")
        self.root.geometry("600x400")
        self.monitor = GPUMonitor()
        self.scheduler = Scheduler()
        self.logger = Logger()
        self.streamer = StreamManager(self.scheduler, self.logger, verbose=False)

        self.create_widgets()
        self.update_dashboard()

    def create_widgets(self):
        self.gpu_frame = ttk.LabelFrame(self.root, text="GPU Status")
        self.gpu_frame.pack(fill="x", padx=10, pady=5)

        self.gpu_labels = {}
        for i in range(2):  # supporte 2 GPUs
            label = ttk.Label(self.gpu_frame, text=f"GPU {i}: ...")
            label.pack(anchor="w", padx=10)
            self.gpu_labels[i] = label

        self.layer_frame = ttk.LabelFrame(self.root, text="Couches chargées")
        self.layer_frame.pack(fill="x", padx=10, pady=5)

        self.layer_list = tk.Listbox(self.layer_frame, height=6)
        self.layer_list.pack(fill="both", padx=10, pady=5)

        self.log_frame = ttk.LabelFrame(self.root, text="Logs")
        self.log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.log_text = tk.Text(self.log_frame, height=8)
        self.log_text.pack(fill="both", padx=10, pady=5)

    def update_dashboard(self):
        # GPU status
        status = self.monitor.status()
        for i, label in self.gpu_labels.items():
            label.config(text=f"GPU {i}: {status.get(f'GPU {i}', 'N/A')}")

        # Couches chargées
        self.layer_list.delete(0, tk.END)
        for name in self.streamer.loaded_layers.keys():
            self.layer_list.insert(tk.END, name)

        # Logs
        logs = self.logger.get_recent_logs()
        self.log_text.delete("1.0", tk.END)
        for log in logs[-10:]:
            self.log_text.insert(tk.END, f"{log}\n")

        self.root.after(1000, self.update_dashboard)  # refresh toutes les secondes

def launch_dashboard():
    root = tk.Tk()
    app = DashboardGUI(root)
    root.mainloop()
