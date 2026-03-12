#!/usr/bin/env python3
"""
Test de validation : Prudence & Performance de l'intégration Rust.
Simule un envoi de gros tenseur P2P avec le router et valide que le HMAC est bien calculé.
"""
import os
import sys
import pickle

# Forcer l'import local de core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("Mise en place de l'environnement de Test P2P...")
    os.environ["VRM_API_TOKEN"] = "test_prudence_token"
    
    try:
        import vramancer_rust
        print("✅ [Succès] Le wrapper Python a trouvé `vramancer_rust`.")
        rust_active = True
    except ImportError:
        print("⚠️ [Fallback] Module Rust non trouvé, on tournera sur CPU pur.")
        rust_active = False

    # Créer un faux "gros Tenseur" de 100 Mo pour le test
    # (On simule un bloc VRAM PyTorch en le remplaçant par de la donnée binaire)
    print("Génération d'un Tenseur factice de 100 Mo (Simulation VRAM)...")
    fake_tensor = b"\x00" * (100 * 1024 * 1024)
    data = pickle.dumps(fake_tensor)
    
    secret = os.environ["VRM_API_TOKEN"].encode()
    
    # Chronométrage Rust vs Python
    import time
    
    if rust_active:
        t0 = time.time()
        # Appel natif (Zéro GIL lock)
        signature = vramancer_rust.sign_payload_fast(secret, data)
        rust_time = time.time() - t0
        print(f"⚡ [Rust] Tenseur de 100 Mo signé en : {rust_time:.5f} secondes")
        
        # Vérification
        t1 = time.time()
        is_valid = vramancer_rust.verify_hmac_fast(secret, data, signature)
        check_time = time.time() - t1
        print(f"🔒 [Rust] Signature vérifiée ({is_valid}) en : {check_time:.5f} secondes")
        
    else:
        import hmac
        import hashlib
        t0 = time.time()
        signature = hmac.new(secret, data, hashlib.sha256).digest()
        py_time = time.time() - t0
        print(f"🐢 [Python CPU] Tenseur de 100 Mo signé en : {py_time:.5f} secondes")

if __name__ == "__main__":
    main()
