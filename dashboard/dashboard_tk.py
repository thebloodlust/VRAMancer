import tkinter as tk
from utils.gpu_utils import get_available_gpus
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def create_live_graph(parent, gpus):
    fig = Figure(figsize=(6, 3), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().pack(pady=10)

    def update():
        ax.clear()
        names = [gpu["name"] for gpu in gpus]
        usage = [gpu["used_vram_mb"] for gpu in gpus]
        total = [gpu["total_vram_mb"] for gpu in gpus]
        percent = [round(u / t * 100, 2) for u, t in zip(usage, total)]
        bars = ax.bar(names, percent, color="#00BFFF")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Utilisation VRAM (%)")
        ax.set_title("Utilisation VRAM en temps réel")
        for bar, p in zip(bars, percent):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{p}%", ha='center', va='bottom')
        canvas.draw()
        parent.after(1000, update)

    update()

def launch_dashboard():
    gpus = get_available_gpus()
    root = tk.Tk()
    root.title("🧠 VRAMancer Dashboard")
    root.configure(bg="#1e1e1e")

    title = tk.Label(root, text="VRAMancer - GPU Monitor", font=("Arial", 18), fg="#00BFFF", bg="#1e1e1e")
    title.pack(pady=10)

    create_live_graph(root, gpus)

    for gpu in gpus:
        frame = tk.Frame(root, bg="#2e2e2e", padx=10, pady=10)
        frame.pack(pady=5, fill="x")

        name = tk.Label(frame, text=f"🎮 {gpu['name']}", font=("Arial", 14), fg="white", bg="#2e2e2e")
        name.pack(anchor="w")

        vram = tk.Label(frame, text=f"💾 VRAM: {gpu['used_vram_mb']} / {gpu['total_vram_mb']} MB", fg="white", bg="#2e2e2e")
        vram.pack(anchor="w")

        status = "✅ Disponible" if gpu["is_available"] else "❌ Indisponible"
        status_label = tk.Label(frame, text=f"📡 Statut: {status}", fg="white", bg="#2e2e2e")
        status_label.pack(anchor="w")

    root.mainloop()
