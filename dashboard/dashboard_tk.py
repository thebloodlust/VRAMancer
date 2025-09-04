import tkinter as tk
from tkinter import ttk
import torch
import time
from utils.gpu_utils import get_available_gpus

def run_real_benchmark(gpu_index):
    torch.cuda.set_device(gpu_index)
    start = time.time()
    size = (1024, 1024)
    a = torch.randn(size, device='cuda')
    b = torch.randn(size, device='cuda')
    for _ in range(1000):
        c = a @ b
    torch.cuda.synchronize()
    end = time.time()
    return round(1000 / (end - start), 2)  # opérations/sec

def launch_dashboard():
    root = tk.Tk()
    root.title("VRAMancer — GPU Dashboard")
    root.configure(bg="#1e1e1e")
    root.geometry("900x700")

    title = tk.Label(root, text="🧠 GPU Monitor", font=("Segoe UI", 24), fg="#00ffcc", bg="#1e1e1e")
    title.pack(pady=20)

    gpus = get_available_gpus()

    for idx, gpu in enumerate(gpus):
        frame = tk.Frame(root, bg="#2e2e2e", bd=2, relief="ridge", padx=10, pady=10)
        frame.pack(pady=10, padx=20, fill="x")

        name = tk.Label(frame, text=f"🎮 {gpu['name']}", font=("Segoe UI", 16), fg="#ffffff", bg="#2e2e2e")
        name.pack(anchor="w")

        vram_text = f"{gpu['used_vram_mb']} / {gpu['total_vram_mb']} MB"
        vram = tk.Label(frame, text=f"💾 VRAM utilisée: {vram_text}", font=("Segoe UI", 12), fg="#cccccc", bg="#2e2e2e")
        vram.pack(anchor="w")

        # Jauge graphique
        percent = int((gpu['used_vram_mb'] / gpu['total_vram_mb']) * 100)
        bar = ttk.Progressbar(frame, length=300, value=percent)
        bar.pack(anchor="w", pady=5)

        status = "✅ Disponible" if gpu["is_available"] else "❌ Indisponible"
        stat = tk.Label(frame, text=f"📡 Statut: {status}", font=("Segoe UI", 12), fg="#cccccc", bg="#2e2e2e")
        stat.pack(anchor="w")

        # Clignotement si VRAM faible
        if gpu["total_vram_mb"] < 2048:
            def blink():
                current = vram.cget("fg")
                vram.config(fg="#ff4444" if current == "#cccccc" else "#cccccc")
                frame.after(500, blink)
            blink()

        # Survol interactif
        def on_enter(e): frame.config(bg="#3e3e3e")
        def on_leave(e): frame.config(bg="#2e2e2e")
        frame.bind("<Enter>", on_enter)
        frame.bind("<Leave>", on_leave)

        # Bouton Benchmark réel
        def benchmark_action(i=idx):
            result = run_real_benchmark(i)
            benchmark_label.config(text=f"🚀 {gpu['name']}: {result} ops/sec")

        btn = tk.Button(frame, text="⚡ Benchmark réel", command=benchmark_action, bg="#00aa88", fg="white", font=("Segoe UI", 12))
        btn.pack(anchor="e", pady=5)

        benchmark_label = tk.Label(frame, text="", font=("Segoe UI", 10), fg="#00ffcc", bg="#2e2e2e")
        benchmark_label.pack(anchor="e")

    footer = tk.Label(root, text="🔧 VRAMancer v1.0 — Système prêt", font=("Segoe UI", 10), fg="#888888", bg="#1e1e1e")
    footer.pack(side="bottom", pady=10)

    root.mainloop()
