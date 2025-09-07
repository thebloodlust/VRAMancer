# dashboard/visualizer.py
import tkinter as tk
from tkinter import ttk
from utils.gpu_utils import get_available_gpus
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use("TkAgg")  # force l’usage du backend Tk

class GpuMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 VRAMancer Live Monitor")
        self.root.configure(bg="#1e1e1e")

        # --- Barre de titre ---
        title = ttk.Label(
            root,
            text="VRAMancer – Suivi GPU en temps réel",
            font=("Arial", 16),
            foreground="#00BFFF",
            background="#1e1e1e",
        )
        title.pack(pady=10)

        # --- Canvas Matplotlib ----
        self.fig, self.ax = plt.subplots(figsize=(6, 3), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)

        # --- Légende détaillée ---
        self.details_frame = ttk.LabelFrame(root, text="Détails")
        self.details_frame.pack(fill=tk.X, padx=10, pady=5)
        self.details_text = tk.Text(self.details_frame, height=8)
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Démarrage de l’actualisation ---
        self.update_dashboard()

    def update_dashboard(self):
        gpus = get_available_gpus()
        self.ax.clear()

        # Barres d’utilisation
        names = [gpu["name"] for gpu in gpus]
        usage = [gpu["used_vram_mb"] for gpu in gpus]
        total = [gpu["total_vram_mb"] for gpu in gpus]
        percent = [round(u / t * 100, 1) for u, t in zip(usage, total)]

        bars = self.ax.bar(names, percent, color="#00BFFF")
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel("Utilisation VRAM (%)")
        self.ax.set_title("Utilisation GPU en temps réel")

        for bar, p in zip(bars, percent):
            self.ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{p}%",
                ha="center",
                va="bottom",
                color="white",
            )

        self.canvas.draw()

        # --- Texte détaillé -------------
        self.details_text.delete("1.0", tk.END)
        for gpu in gpus:
            status = "✅ Disponible" if gpu["is_available"] else "❌ Indisponible"
            self.details_text.insert(
                tk.END,
                f"🎮 {gpu['name']}\n"
                f"💾 {gpu['used_vram_mb']} / {gpu['total_vram_mb']} MB\n"
                f"📡 {status}\n\n",
            )

        # On rafraîchit toutes les 2 s
        self.root.after(2000, self.update_dashboard)

def launch_dashboard():
    root = tk.Tk()
    root.geometry("700x500")
    app = GpuMonitorGUI(root)
    root.mainloop()
