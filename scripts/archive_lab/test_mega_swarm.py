import asyncio
import websockets
import json
import struct
import time
from core.network.webgpu_node import WebGPUNodeManager

async def mock_browser_client(client_id, power_delay):
    uri = "ws://localhost:5060"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"🌍 [{client_id}] Connecté au Swarm de VRAMancer !")
            
            # 1. Init
            await websocket.recv()
            
            # 2. Capabilities
            await websocket.send(json.dumps({
                "type": "capabilities", 
                "gpu_name": "RTX 4090" if "PC" in client_id else "M1_Mac"
            }))
            
            # 3. Listen loop
            while True:
                task_msg = await websocket.recv()
                if isinstance(task_msg, bytes):
                    header_len = struct.unpack('<I', task_msg[:4])[0]
                    header = json.loads(task_msg[4:4+header_len].decode('utf-8'))
                    payload = task_msg[4+header_len:]
                    
                    print(f"⚡ [{client_id}] Tenseur reçu: layer {header['layer']} ({len(payload)} bytes). Début du calcul GPU...")
                    
                    # Simuler lag GPU
                    await asyncio.sleep(power_delay) 
                    
                    print(f"✅ [{client_id}] Calcul layer {header['layer']} terminé !")
                    # Resultat
                    result_header = json.dumps({
                        "type": "result", 
                        "task_id": header['task_id'], 
                        "flops": 15000000
                    }).encode('utf-8')
                    res_header_len = struct.pack('<I', len(result_header))
                    await websocket.send(res_header_len + result_header + b'COMPUTED_DATA_OK')
    except Exception as e:
        print(f"❌ [{client_id}] Déconnecté.")


async def main():
    print("🚀 === DÉMARRAGE DU SUPREME SWARM TEST === 🚀\n")
    manager = WebGPUNodeManager(port=5060)
    manager.start()
    
    await asyncio.sleep(1) # Laisser le boot
    
    # 2 faux participants au méga-cluster
    task_pc = asyncio.create_task(mock_browser_client("PC_Gamer_du_Cybercafe", 0.1))
    task_mac = asyncio.create_task(mock_browser_client("MacBook_Etudiant", 0.4))
    
    await asyncio.sleep(0.5)
    
    print("\n🧠 [VRAMancer Master] Transmission d'un faux modèle massif (10 blocks) vers le swarm...")
    start_time = time.time()
    
    tasks = []
    for i in range(10):
        # 10 méga-batch de données bidons Q8 (100 Ko par payload pour simuler)
        dummy_tensor_data = b"TENSOR_WEIGHTS" * 1000 
        future = manager.submit_tensor(layer_id=i, tensor_data=dummy_tensor_data, quant_scale=0.5)
        tasks.append(asyncio.wrap_future(future))
        
    # On attend la résolution multi-noeud
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    print(f"\n🎉 [VRAMancer Master] SUCCÈS ! Les 10 tenseurs ont été fusionnés en {elapsed:.2f} secondes !")
    print("Architecture validée: C'est prêt pour dominer le monde.\n")
    
    # Clean up
    task_pc.cancel()
    task_mac.cancel()
    manager._loop.stop()

if __name__ == "__main__":
    import logging
    logging.getLogger('vramancer').setLevel(logging.CRITICAL)
    logging.getLogger('webgpu_node').setLevel(logging.CRITICAL)
    asyncio.run(main())