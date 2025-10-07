#!/usr/bin/env python3
"""Tkinter Dashboard VRAMancer - Version simplifiée."""

import sys
import os

def main():
    print("=" * 60)
    print("      VRAMANCER TKINTER DASHBOARD")
    print("=" * 60)
    print()
    
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        print("✓ Tkinter disponible")
        
        # Test requests
        try:
            import requests
            print("✓ Requests disponible")
        except ImportError:
            print("❌ Requests manquant - installation...")
            os.system("pip install requests")
        
        print("Création de l'interface Tkinter...")
        
        # Fenêtre principale
        root = tk.Tk()
        root.title("VRAMancer Dashboard - Tkinter Simple")
        root.geometry("600x400")
        
        # Frame principal
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titre
        title_label = ttk.Label(main_frame, text="VRAMancer Dashboard", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Infos
        info_text = tk.Text(main_frame, height=15, width=70, wrap=tk.WORD)
        info_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Message initial
        info_text.insert(tk.END, "=== VRAMANCER TKINTER DASHBOARD ===\n\n")
        info_text.insert(tk.END, "✓ Interface Tkinter initialisée\n")
        info_text.insert(tk.END, "✓ Fenêtre graphique créée\n")
        info_text.insert(tk.END, "✓ Composants UI chargés\n\n")
        
        # Test API
        try:
            import requests
            response = requests.get("http://localhost:5030/health", timeout=3)
            if response.status_code == 200:
                info_text.insert(tk.END, "✓ API VRAMancer accessible\n")
                info_text.insert(tk.END, "✓ Connexion établie sur port 5030\n")
            else:
                info_text.insert(tk.END, f"⚠️  API Status: {response.status_code}\n")
        except:
            info_text.insert(tk.END, "❌ API VRAMancer non accessible\n")
            info_text.insert(tk.END, "   Lancez l'API avec option 1 du menu\n")
        
        info_text.insert(tk.END, "\n=== INTERFACE FONCTIONNELLE ===\n")
        info_text.insert(tk.END, "Cette fenêtre démontre que Tkinter fonctionne.\n")
        info_text.insert(tk.END, "Pour une interface complète, utilisez l'option 11 (Qt Dashboard).\n")
        
        # Boutons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        def close_app():
            root.destroy()
        
        def show_info():
            messagebox.showinfo("VRAMancer", "Interface Tkinter fonctionnelle!\nUtilisez l'option 11 pour l'interface complète.")
        
        ttk.Button(button_frame, text="Test Message", command=show_info).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Fermer", command=close_app).pack(side=tk.RIGHT)
        
        print("✓ Interface Tkinter créée")
        print("Une fenêtre graphique devrait s'ouvrir...")
        print("Fermeture automatique dans 15 secondes si pas d'interaction")
        
        # Auto-fermeture après 15 secondes
        root.after(15000, close_app)
        
        # Lancement
        root.mainloop()
        
        print("Interface Tkinter fermée")
        
    except ImportError as e:
        print(f"❌ Erreur import: {e}")
        print("Tkinter devrait être inclus avec Python...")
    except Exception as e:
        print(f"❌ Erreur Tkinter: {e}")

if __name__ == "__main__":
    main()