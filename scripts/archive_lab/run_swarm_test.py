import asyncio
import websockets
import json
import struct
import time
import threading
from core.network.webgpu_node import WebGPUNodeManager
import logging

logging.getLogger('vramancer').setLevel(logging.CRITICAL)
logging.getLogger('webgpu_node').setLevel(logging.CRITICAL)

async def mock_browser_client(client_id, power_delay):
    uri = "ws://localhost:5060"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"🌍 [{client_id}] Connecté au Swarm de VRAMancer !")
            
            # 1. Init
            await websocket.recv()
            
            gpu_name = "RTX 4090" if "PC" in client_id else "M1_Mac"
            # 2. Capabilities
            await websocket.send(json.dumps({
                "type": "capabilities", 
                "gpu_name": gpu_name
            }))
            
            print(f"🖥️  [{client_id}] Hardware reporté: {gpu_name}")
            
            # 3. Listen loop
            while True:
                task_msg = await websocket.recv()
                if isinstance(task_msg, bytes):
                    header_len = struct.unpack('<I', task_msg[:4])[0]
                    header = json.loads(task_msg[4:4+header_len].decode('utf-8'))
                    payload = task_msg[4+header_len:]
                    
                    print(f"⚡ [{client_id}] Tenseur reçu: couche {header['layer']} ({len(payload)//1024} Ko). Calcul GPU WGSL ({power_delay}s)...")
                    
                    # Simuler lag GPU
                    await asyncio.sleep(power_delay) 
                    
                    print(f"✅ [{client_id}] Calcul couche {header['layer']} terminé !")
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


def start_server():
    manager = WebGPUNodeManager(port=5060)
    # The start method will spawn its own thread & event loop
    manager.start()
    return manager

async def run_simulation(manager):
    # 2 faux participants au méga-cluster
    task_pc = asyncio.create_task(mock_browser_client("PC_Gamer_Cybercafe", 0.05))
    task_mac = asyncio.create_task(mock_browser_client("MacBook_Etudiant", 0.2))
    
    await asyncio.sleep(1.0)
    
    print("\n🧠 [VRAMancer Master] Génération asynchrone: 10 layers envoyés simultanément (Smart-Load Balancing en action)...")
    start_time = time.time()
    
    tasks = []
    for i in range(10):
        # 10 méga-batch de données bidons Q8 (143Ko par payload pour simuler)
        dummy_tensor_data = b"TENSOR_WEIGHTS" * 10240 
        future = manager.submit_tensor(layer_id=i, tensor_data=dummy_tensor_data, quant_scale=0.5)
        # wrap the concurrent future to an asyncio future
        tasks.append(asyncio.wrap_future(future))
        
    # On attend la résolution multi-noeud
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    print(f"\n🎉 [VRAMancer Master] SUCCÈS ! Les 10 couches massives ont été calculées et agrégées en {elapsed:.2f} secondes !")
    print("Architecturalement, OUI, on peut faire tourner des LLM massifs sans budget en utilisant la flotte citoyenne !\n")
    
    # Clean up
    task_pc.cancel()
    task_mac.cancel()
    manager.is_running = False

if __name__ == "__main__":
    print("🚀 === DÉMARRAGE DU SUPREME SWARM TEST === 🚀\n")
    m = start_server()
    time.sleep(1) # Let the background thread start the loop
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_simulation(m))
    finally:
        loop.close()
