@echo off
echo ============================================================
echo   VRAMANCER - CORRECTION IMPORTS PYTHON
echo ============================================================

echo.
echo === CORRECTION CHEMIN PYTHON ===
cd /d "C:\Users\the_b\Desktop\VRAMancer-main\VRAMancer-main"
set PYTHONPATH=%cd%

echo ✅ PYTHONPATH configuré: %PYTHONPATH%
echo.

echo === TEST DASHBOARD TKINTER (corrigé) ===
python -c "
import sys
sys.path.insert(0, '.')
try:
    from dashboard.visualizer import launch_tk_dashboard
    print('✅ Import dashboard.visualizer OK')
    launch_tk_dashboard()
except ImportError as e:
    print('❌ Import error:', e)
    print('🔧 Lancement direct...')
    import tkinter as tk
    from tkinter import ttk
    import requests
    
    def create_tk_dashboard():
        root = tk.Tk()
        root.title('VRAMancer Dashboard')
        root.geometry('800x600')
        
        # Test API
        try:
            resp = requests.get('http://localhost:5030/health')
            status = '✅ API OK' if resp.status_code == 200 else '❌ API Error'
        except:
            status = '❌ API Non connectée'
        
        label = tk.Label(root, text=f'VRAMancer Dashboard\n{status}', font=('Arial', 16))
        label.pack(pady=20)
        
        quit_btn = tk.Button(root, text='Quitter', command=root.quit)
        quit_btn.pack(pady=10)
        
        root.mainloop()
    
    create_tk_dashboard()
"

pause