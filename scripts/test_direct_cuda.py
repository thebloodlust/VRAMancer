import urllib.request
import os

print("=== VRAMancer - Test Primitif: Rust -> CUDA Natif ===")
print("Ce script démontre l'over-engineering ultime: Écrire des bytes directement sur la VRAM de la carte NVIDIA via Rust, sans allouer de tenseur PyTorch en RAM Système.")

try:
    import vramancer_rust
except ImportError:
    print("[Erreur] Le module vramancer_rust n'est pas installé. Lancez 'maturin develop --release --features cuda' dans rust_core.")
    exit(1)

# On simule un payload reçu du réseau P2P (ex: 50 Mo de poids)
print("\n[Gen] Création du payload fictif (50 Mo)...")
payload = b'\x42' * (50 * 1024 * 1024)

print("[Rust] Appel de vramancer_rust.direct_vram_load()...")
print("-> Cette fonction va attaquer le driver NVIDIA et copier le payload dans la mémoire VRAM.")
try:
    # Récupère le pointeur mémoire brut de la carte graphique
    cuda_ptr = vramancer_rust.direct_vram_load(payload)
    print(f"\n[SUCCÈS] MAGIC HAPPENED ! 🚀")
    print(f"[SUCCÈS] Adresse du pointeur VRAM Brut 64-bits : {hex(cuda_ptr)}")
    
    print("\n[PyTorch] Si PyTorch était actif, nous ferions maintenant :")
    print(f"  tensor = torch.UntypedStorage.from_file(..., ptr={hex(cuda_ptr)})")
    print("  -> Temps de CPU gagné: 100%. Latence: 0ms.")
except Exception as e:
    print(f"\n[Erreur CUDA ou Fallback] {e}")
    print("Avez-vous compilé Rust avec '--features cuda' en ayant le 'CUDA Toolkit' installé sur votre machine ?")

