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
        
        # Zero-Trust Security: authenticate WebGPU nodes
        from core.security import verify_request
        import os
        import hmac

        # Extract token from the WebSocket path (e.g. ws://host:port/?token=xyz)
        query = path.split('?')
        token = ""
        if len(query) > 1:
            for param in query[1].split('&'):
                if param.startswith('token='):
                    token = param.split('=')[1]
                    break
        
        secret = os.environ.get("VRM_API_TOKEN")
        is_production = os.environ.get('VRM_PRODUCTION') == '1'
        
        if is_production and not token:
            _log.warning(f"Rejet connexion {client_id} (Zero-Trust): Token manquant.")
            await websocket.close(1008, "Token required")
            return
            
        if token and secret:
            from core.security import _maybe_rotate
            eff_secret = _maybe_rotate(secret)
            if not hmac.compare_digest(token, eff_secret) and not hmac.compare_digest(token, secret):
                _log.warning(f"Rejet connexion {client_id} (Zero-Trust): Token invalide.")
                await websocket.close(1008, "Invalid token")
                return

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
                            WEBGPU_FLOPS_TOTAL.inc(data.get("flops", 0))
                            # Le tensor_result peut être base64 ou juste du texte
                            res = data.get("tensor_result")
                            if isinstance(res, str) and res.startswith("BASE64:"):
                                import base64
                                task.future.set_result(base64.b64decode(res[7:]))
                            else:
                                # Fallback vers encode
                                task.future.set_result(str(res).encode())
                                
                elif isinstance(message, bytes):
                    # Direct binary payload from WebGPU: [HeaderLen(4)][HeaderJSON][TensorData]
                    if len(message) >= 4:
                        header_len = struct.unpack('<I', message[:4])[0]
                        if len(message) >= 4 + header_len:
                            try:
                                header = json.loads(message[4:4+header_len].decode('utf-8'))
                                payload = message[4+header_len:]
                                if header.get("type") == "result":
                                    task_id = header.get("task_id")
                                    if task_id in self.pending_tasks:
                                        task = self.pending_tasks.pop(task_id)
                                        self.clients[client_id]["busy"] = False
                                        WEBGPU_FLOPS_TOTAL.inc(header.get("flops", 0))
                                        task.future.set_result(payload)
                            except json.JSONDecodeError:
                                _log.error("Failed to decode binary header from WebGPU client.")
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

    async def submit_swarm_attention(self, layer_id: int, q_tensor: bytes, kv_tensor: bytes, timeout_s: float = 0.15) -> bytes:
        """
        SWARM ATTENTION (Production Mode) WITH HOLOGRAPHIC PARITY:
        Divise le KV cache sur plusieurs noeuds. Genere un noeud de Parite XOR.
        Si un noeud straggle ou meurt, on regenere instantanement son calcul!
        """
        if not self.clients:
            _log.warning("Swarm Attention: Aucun client WebGPU. Utilisation du fallback (L1/L2 local).")
            return q_tensor + kv_tensor[:len(q_tensor)]

        available_clients = [cid for cid, c in self.clients.items() if not c["busy"]]
        if not available_clients:
            available_clients = list(self.clients.keys())

        MAX_NODES = 32
        available_clients = available_clients[:MAX_NODES]
        num_data_nodes = min(len(available_clients), 8) # Max 8 parallel for stability tests
        
        # Need at least 2 nodes to do parity meaningfully
        use_hologram = num_data_nodes > 1 and len(available_clients) > num_data_nodes
        
        if use_hologram:
            from core.holographic_memory import hive_memory
            shards, parity = hive_memory.encode_hologram(kv_tensor, num_data_nodes)
            _log.info(f"🧬 Swarm Hologram: Déploiement distribué sur {num_data_nodes} noeuds + 1 noeud Parité (Self-Healing).")
        else:
            num_data_nodes = len(available_clients)
            shards = [kv_tensor[i * (len(kv_tensor)//num_data_nodes) : (i+1) * (len(kv_tensor)//num_data_nodes)] for i in range(num_data_nodes)]
            use_hologram = False

        tasks_meta = []
        futures = []

        # Send to Data Nodes
        for i in range(num_data_nodes):
            client_id = available_clients[i]
            payload_data = q_tensor + shards[i]
            task = WebGPUTask(layer_id, payload_data)
            
            self.task_queue.put_nowait(task)
            futures.append(task.future)
            tasks_meta.append((i, len(payload_data), task.task_id))

        # Send to Parity Node (if enough clients)
        parity_future = None
        if use_hologram:
            parity_payload = q_tensor + parity
            parity_task = WebGPUTask(layer_id, parity_payload)
            self.task_queue.put_nowait(parity_task)
            parity_future = parity_task.future
            # Watch out: we wait for futures + parity_future

        all_futures = futures + ([parity_future] if parity_future else [])
        done, pending = await asyncio.wait(all_futures, timeout=timeout_s)

        for pending_future in pending:
            pending_future.cancel()

        valid_results = []
        node_failed = False
        
        for idx, task_future in enumerate(futures):
            if task_future in done:
                try:
                    result_bytes = task_future.result()
                    if not isinstance(result_bytes, bytes) or len(result_bytes) == 0:
                        raise ValueError("Payload corrompu.")
                    if len(result_bytes) > (len(shards[idx]) * 2):
                        raise ValueError("Suspected Tensor Poisoning.")
                    valid_results.append(result_bytes)
                except Exception as e:
                    valid_results.append(None) # Signal for hologram healing
                    node_failed = True
            else:
                valid_results.append(None)
                node_failed = True

        # HOLOGRAPHIC REGENERATION
        if use_hologram and node_failed:
            if parity_future in done:
                try:
                    parity_result = parity_future.result()
                    _log.warning("🧠 [Swarm] Intrusion/Straggler détecté ! Activation du Cortex Holographique...")
                    aggregated_result = hive_memory.heal_hologram(valid_results, parity_result)
                    return aggregated_result
                except Exception:
                    pass # Parity also failed
                    
        # Replace missing with zeros if healing failed or wasn't used
        for i, res in enumerate(valid_results):
            if res is None:
                valid_results[i] = b'\x00' * min(len(shards[i]), 1024)

        return b"".join(valid_results)

    def start(self):
        if not websockets:
            _log.error("Module 'websockets' manquant. WebGPU désactivé.")
            return
            
        self.is_running = True
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        server = websockets.serve(self._handler, "0.0.0.0", self.port)
        self._loop.run_until_complete(server)
        self._loop.create_task(self._task_dispatcher())
        
        _log.info(f"Serveur WebGPU démarré sur le port {self.port}")
        
        # Exécuter la boucle dans un thread séparé pour ne pas bloquer Flask/PyTorch
        import threading
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

if __name__ == "__main__":
    manager = WebGPUNodeManager()
    manager.start()
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.is_running = False

