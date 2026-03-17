import time
import asyncio
from core.network.webgpu_node import WebGPUNodeManager, WebGPUTask

async def main():
    manager = WebGPUNodeManager(port=8560)
    manager.start()
    
    print("==================================================")
    print("🟢 MASTER NODE DEMARRE")
    print("👉 CONNECTEZ LE MOBILE DEPUIS L'APP WEB MAINTENANT")
    print("==================================================")
    
    while len(manager.clients) == 0:
        await asyncio.sleep(1)
        
    client_id = list(manager.clients.keys())[0]
    print(f"\n📡 Mobile détecté dans le Swarm ! (ID: {client_id})")
    print("⏳ Attente de la fin du handshake (2s)...")
    await asyncio.sleep(2)
    
    print("\n🚀 [SCHEDULER] Création d'une tâche de calcul (Layer 42, 1KB de Tenseur mocké)...")
    fake_tensor = b"\x00\xFF" * 500
    
    start_time = time.time()
    
    print(f"📦 [NETWORK] Envoi du tenseur vers le Node {client_id} en WebSockets/Binaire...")
    
    # Injection dans le dispatcher !
    future = manager.dispatch_layer(layer_id=42, tensor_data=fake_tensor)
    
    print("⏳ [MASTER] Tâche envoyée. En attente que le téléphone fasse l'inférence WebGPU...")
    
    result = await future
    
    elapsed = time.time() - start_time
    print(f"\n✅ [SUCCÈS] Le mobile a renvoyé le tenseur calculé !")
    print(f"⏱️  Temps total d'aller-retour (Round-trip) : {elapsed:.3f} s")
    print(f"📊 Extrait du tenseur de sortie : {str(result)[:60]}...")
    print("==================================================\n")
    
if __name__ == "__main__":
    asyncio.run(main())
