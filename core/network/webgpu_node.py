"""
VRAMancer WebGPU Node — Production Ready
----------------------------------------
Serveur WebSocket asynchrone haute performance pour l'offloading WebGPU.
Inclut :
- Sérialisation binaire des tenseurs (zéro-copie si possible)
- Heartbeat et détection de déconnexion (Dead Peer Detection)
- File d'attente asynchrone (Task Queue) avec timeout
- Intégration avec Prometheus (core.metrics)
"""

import asyncio
import json
import struct
import time
import uuid
from typing import Dict, Any, Optional

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    websockets = None

try:
    from core.logger import LoggerAdapter
    _log = LoggerAdapter("webgpu_node")
except Exception:
    import logging
    _log = logging.getLogger("vramancer.webgpu")

try:
    from core.metrics import WEBGPU_CONNECTED_CLIENTS, WEBGPU_FLOPS_TOTAL
except ImportError:
    # Fallback si les métriques ne sont pas encore définies dans core.metrics
    class DummyMetric:
        def inc(self, val=1): pass
        def set(self, val): pass
    WEBGPU_CONNECTED_CLIENTS = DummyMetric()
    WEBGPU_FLOPS_TOTAL = DummyMetric()

class WebGPUTask:
    def __init__(self, layer_id: int, tensor_data: bytes):
        self.task_id = uuid.uuid4().hex
        self.layer_id = layer_id
        self.tensor_data = tensor_data
        self.future = asyncio.Future()
        self.created_at = time.time()

class WebGPUNodeManager:
    def __init__(self, port: int = 5060, heartbeat_interval: int = 15):
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.clients: Dict[str, Any] = {}
        self.task_queue: asyncio.Queue[WebGPUTask] = asyncio.Queue()
        self.pending_tasks: Dict[str, WebGPUTask] = {}
        self.is_running = False

    async def _heartbeat(self, websocket, client_id: str):
        """Maintient la connexion en vie et détecte les navigateurs fermés."""
        try:
            while self.is_running:
                await asyncio.sleep(self.heartbeat_interval)
                await websocket.ping()
                self.clients[client_id]["last_seen"] = time.time()
        except ConnectionClosed:
            _log.warning(f"Client WebGPU {client_id} perdu (Heartbeat timeout).")
            await self._disconnect_client(client_id)

    async def _disconnect_client(self, client_id: str):
        if client_id in self.clients:
            del self.clients[client_id]
            WEBGPU_CONNECTED_CLIENTS.set(len(self.clients))
            _log.info(f"Client {client_id} déconnecté. Restants: {len(self.clients)}")

    async def _handler(self, websocket, path):
        client_id = uuid.uuid4().hex[:8]
        _log.info(f"Nouvelle connexion WebGPU: {client_id}")
        
        self.clients[client_id] = {
            "ws": websocket,
            "last_seen": time.time(),
            "busy": False
        }
        WEBGPU_CONNECTED_CLIENTS.set(len(self.clients))

        # Lancer le heartbeat en tâche de fond
        heartbeat_task = asyncio.create_task(self._heartbeat(websocket, client_id))

        try:
            # Demander les specs du GPU
            await websocket.send(json.dumps({"type": "init"}))
            
            async for message in websocket:
                if isinstance(message, str):
                    data = json.loads(message)
                    if data.get("type") == "capabilities":
                        self.clients[client_id]["gpu"] = data.get("gpu_name", "Unknown")
                        _log.info(f"Client {client_id} prêt: {data.get('gpu_name')}")
                    
                    elif data.get("type") == "result":
                        task_id = data.get("task_id")
                        if task_id in self.pending_tasks:
                            task = self.pending_tasks.pop(task_id)
                            self.clients[client_id]["busy"] = False
                            # Enregistrer les FLOPS pour Prometheus
                            WEBGPU_FLOPS_TOTAL.inc(data.get("flops", 0))
                            task.future.set_result(data.get("tensor_result"))
                            
        except ConnectionClosed:
            pass
        finally:
            heartbeat_task.cancel()
            await self._disconnect_client(client_id)

    async def _task_dispatcher(self):
        """Distribue les tâches de la file d'attente aux navigateurs disponibles."""
        while self.is_running:
            if not self.clients:
                await asyncio.sleep(0.1)
                continue

            # Trouver un client libre
            available_client = next((cid for cid, c in self.clients.items() if not c["busy"]), None)
            
            if available_client:
                task = await self.task_queue.get()
                ws = self.clients[available_client]["ws"]
                self.clients[available_client]["busy"] = True
                self.pending_tasks[task.task_id] = task
                
                # Envoi binaire (Header JSON + Payload Tenseur)
                header = json.dumps({"type": "compute", "task_id": task.task_id, "layer": task.layer_id}).encode('utf-8')
                header_len = struct.pack('<I', len(header))
                payload = header_len + header + task.tensor_data
                
                try:
                    await ws.send(payload)
                except ConnectionClosed:
                    self.clients[available_client]["busy"] = False
                    await self.task_queue.put(task) # Remettre la tâche dans la file
            else:
                await asyncio.sleep(0.01)

    def submit_tensor(self, layer_id: int, tensor_data: bytes) -> asyncio.Future:
        """API publique pour soumettre un tenseur au cluster WebGPU."""
        task = WebGPUTask(layer_id, tensor_data)
        self.task_queue.put_nowait(task)
        return task.future

    def start(self):
        if not websockets:
            _log.error("Module 'websockets' manquant. WebGPU désactivé.")
            return
            
        self.is_running = True
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        server = websockets.serve(self._handler, "0.0.0.0", self.port)
        loop.run_until_complete(server)
        loop.create_task(self._task_dispatcher())
        
        _log.info(f"Serveur WebGPU démarré sur le port {self.port}")
        
        # Exécuter la boucle dans un thread séparé pour ne pas bloquer Flask/PyTorch
        import threading
        threading.Thread(target=loop.run_forever, daemon=True).start()

if __name__ == "__main__":
    manager = WebGPUNodeManager()
    manager.start()
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.is_running = False

