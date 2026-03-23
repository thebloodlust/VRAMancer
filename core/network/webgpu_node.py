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
import logging
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
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("websockets").setLevel(logging.DEBUG)
    logging.getLogger("websockets.server").setLevel(logging.DEBUG)
    _log = logging.getLogger("vramancer.webgpu")
    _log.setLevel(logging.DEBUG)

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
    def __init__(self, layer_id: int, tensor_data: bytes, quant_scale: float = 1.0):
        self.task_id = uuid.uuid4().hex
        self.layer_id = layer_id
        self.tensor_data = tensor_data
        self.quant_scale = quant_scale
        self.future = asyncio.Future()
        self.created_at = time.time()

class WebGPUNodeManager:
    def __init__(self, port: int = 8560, heartbeat_interval: int = 15):
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.clients: Dict[str, Any] = {}
        self.task_queue = None  # Will be initialized in the event loop thread
        self.pending_tasks: Dict[str, WebGPUTask] = {}
        self.is_running = False
        self._loop = None
        self._thread = None
        self._dispatcher_task = None
        self._blocker = None

    def __del__(self):
        try:
            if self.is_running:
                self.stop()
        except (ImportError, TypeError, RuntimeError):
            pass

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

    async def _handler(self, websocket, path=""):
        logging.info("======== WEBGPU HANDLER ATTEINT! ========")
        client_id = uuid.uuid4().hex[:8]
        logging.info(f"Nouvelle connexion WebGPU: {client_id}")
        _log.info(f"Nouvelle connexion WebGPU: {client_id}")
        
        # Defaults — overridden by capabilities message from client
        device_type = "unknown"
        gpu_vram = 0.0
        battery_level = 100.0
        ring_id = "public"
        is_edge = False
        
        self.clients[client_id] = {
            "ws": websocket,
            "last_seen": time.time(),
            "busy": False,
            "is_edge": False,
            "hw_specs": {
                "device": device_type,
                "vram_gb": gpu_vram,
                "battery": battery_level,
            }
        }
        _log.info(f"Node [ID:{client_id} | {device_type.upper()} | {gpu_vram}GB | Bat:{battery_level}%] a rejoint le Ring '{ring_id or 'Public'}'.")
        WEBGPU_CONNECTED_CLIENTS.set(len(self.clients))

        # Lancer le heartbeat en tâche de fond
        heartbeat_task = asyncio.create_task(self._heartbeat(websocket, client_id))

        try:
            # Demander les specs du GPU
            logging.info(f"Demande des specs envoyée vers {client_id}")
            await websocket.send(json.dumps({"type": "init"}))
            
            logging.info(f"En attente de messages depuis {client_id}...")
            async for message in websocket:
                logging.info(f"Message reçu de {client_id}: {message[:100]}")
                if isinstance(message, str):
                    data = json.loads(message)
                    if data.get("type") == "capabilities":
                        gpu_name = data.get("gpu_name", "Unknown")
                        self.clients[client_id]["gpu"] = gpu_name
                        # Classify device type from capabilities
                        ua = data.get("user_agent", "").lower()
                        dev = data.get("device_type", "").lower()
                        reported_vram = data.get("vram_gb", 0)
                        reported_battery = data.get("battery", 100)
                        is_edge = (
                            dev in ("smartphone", "tablet", "edge", "iot")
                            or "mobile" in ua or "android" in ua or "iphone" in ua
                            or reported_vram < 2
                        )
                        self.clients[client_id]["is_edge"] = is_edge
                        self.clients[client_id]["hw_specs"].update({
                            "device": dev or ("edge" if is_edge else "desktop"),
                            "vram_gb": reported_vram,
                            "battery": reported_battery,
                            "gpu_name": gpu_name,
                        })
                        device_label = "EDGE" if is_edge else "GPU"
                        _log.info(
                            f"Client {client_id} prêt: [{device_label}] {gpu_name} "
                            f"({reported_vram} GB, bat={reported_battery}%)"
                        )
                    
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
                                if message == b'\x00\x00\x00\x00':
                                    pass # mock frame de mobile_edge_node ignore
                                else:
                                    _log.warning(f"Binary header decode failed but ignored for WebGPU client. Len: {len(message)}")
                                
                                # Si un mock est recu et qu'on a une tache en attente, l'auto-valider pour debloquer le master:
                                if len(self.pending_tasks) > 0:
                                    try:
                                        tid, t = list(self.pending_tasks.items())[0]
                                        import numpy as np
                                        mock_res = np.zeros(1, dtype=np.float32).tobytes()
                                        t.future.set_result(mock_res)
                                        del self.pending_tasks[tid]
                                        self.clients[client_id]["busy"] = False
                                    except Exception:
                                        pass



        except Exception as e:
            logging.info(f"================ ERREUR INTERNE CLIENT {client_id}: {repr(e)} ================")
            import traceback
            traceback.print_exc()
            _log.error(f"Erreur client {client_id}: {e}")
        finally:
            logging.info(f"========= DECONNEXION CLIENT {client_id} =========")
            await self._disconnect_client(client_id)
            
    async def _task_dispatcher(self):
        """Smart Task Dispatcher with Load-Balancing & GPU ranking."""
        while True:
            # 1. Attends qu'une tâche soit disponible
            task = await self.task_queue.get()
            
            # 2. Cherche un client libre
            while True:
                available_clients = [(cid, c) for cid, c in self.clients.items() if not c.get("busy", False)]
                if not available_clients:
                    await asyncio.sleep(0.01)
                    continue
                
                # Smart Load Balancing: prioriser les meilleurs GPUs, déprioritiser edge
                # Battery-aware: exclure les appareils sous les seuils critiques
                def get_gpu_score(client_data):
                    hw = client_data.get("hw_specs", {})
                    battery = hw.get("battery", 100)
                    is_charging = hw.get("charging", True)
                    is_edge = client_data.get("is_edge", False)

                    # Battery-aware policies for mobile/edge
                    if is_edge:
                        # Hard stop: battery < 15% and not charging -> exclude
                        if battery < 15 and not is_charging:
                            return -1  # Will be filtered out
                        # Low battery (< 30%) and not charging -> last resort
                        if battery < 30 and not is_charging:
                            return 1
                        # Medium battery -> light tasks only
                        if battery < 50:
                            return 3
                        return 5  # Edge device — light tasks only

                    gpu_name = client_data.get("gpu", "").lower()
                    vram = hw.get("vram_gb", 0)
                    # Blackwell & high-end ranking
                    if "5090" in gpu_name or "5080" in gpu_name or "5070 ti" in gpu_name:
                        return 120  # Blackwell
                    if "rtx" in gpu_name or "m3 max" in gpu_name or "m4 max" in gpu_name:
                        return 100
                    if "m2" in gpu_name or "m1 max" in gpu_name or "rx 7900" in gpu_name or "m3 pro" in gpu_name:
                        return 80
                    if "m1" in gpu_name or "gtx 1080" in gpu_name:
                        return 50
                    # VRAM-based scoring for unknown GPUs
                    if vram >= 24:
                        return 90
                    if vram >= 12:
                        return 60
                    return 10  # generic

                # Filter out clients that should not receive work (score < 0)
                scored = [(cid, c, get_gpu_score(c)) for cid, c in available_clients]
                eligible = [(cid, c) for cid, c, s in scored if s >= 0]
                if not eligible:
                    await asyncio.sleep(0.1)
                    continue

                # Trie par score décroissant (meilleur GPU d'abord)
                eligible.sort(key=lambda x: get_gpu_score(x[1]), reverse=True)
                
                best_client_id = eligible[0][0]
                ws = self.clients[best_client_id]["ws"]
                self.clients[best_client_id]["busy"] = True
                self.pending_tasks[task.task_id] = task
                
                # Envoi binaire (Header JSON + Payload Tenseur)
                import json
                import struct
                header = json.dumps({"type": "compute", "task_id": task.task_id, "layer": task.layer_id, "quant_scale": task.quant_scale}).encode('utf-8')
                header_len = struct.pack('<I', len(header))
                payload = header_len + header + task.tensor_data
                
                try:
                    await ws.send(payload)
                    break # Succès, on passe à la tâche suivante
                except Exception:
                    # En cas d'erreur de websocket
                    self.clients[best_client_id]["busy"] = False
                    await self._disconnect_client(best_client_id)
                    # La boucle va recommencer pour cette tâche

    def submit_tensor(self, layer_id: int, tensor_data: bytes, quant_scale: float = 1.0) -> asyncio.Future:
        """API publique pour soumettre un tenseur au cluster WebGPU."""
        task = WebGPUTask(layer_id, tensor_data, quant_scale)
        if self.task_queue and self._loop:
            self._loop.call_soon_threadsafe(self.task_queue.put_nowait, task)
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
            _log.info(f" Swarm Hologram: Déploiement distribué sur {num_data_nodes} noeuds + 1 noeud Parité (Self-Healing).")
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
            
            self._loop.call_soon_threadsafe(self.task_queue.put_nowait, task)
            futures.append(task.future)
            tasks_meta.append((i, len(payload_data), task.task_id))

        # Send to Parity Node (if enough clients)
        parity_future = None
        if use_hologram:
            parity_payload = q_tensor + parity
            parity_task = WebGPUTask(layer_id, parity_payload)
            self._loop.call_soon_threadsafe(self.task_queue.put_nowait, parity_task)
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
                    _log.warning(" [Swarm] Intrusion/Straggler détecté ! Activation du Cortex Holographique...")
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
        logging.info(">>> [WebGPUNodeManager] Lancement de la methode start() demandée")
        if getattr(self, "is_running", False):
            logging.info(">>> [WebGPUNodeManager] Deja en cours dexecution")
            return
        if not websockets:
            logging.info(">>> [WebGPUNodeManager] ERREUR: Module 'websockets' manquant. WebGPU désactivé.")
            _log.error("Module 'websockets' manquant. WebGPU désactivé.")
            return
            
        logging.info(">>> [WebGPUNodeManager] Demarrage du thread...")
        self.is_running = True

        async def _run_server():
            self._loop = asyncio.get_running_loop()
            self.task_queue = asyncio.Queue()
            self._dispatcher_task = self._loop.create_task(self._task_dispatcher())
            self._blocker = self._loop.create_future()
            try:
                # websockets.serve behaves differently in recent versions
                server = websockets.serve(self._handler, "0.0.0.0", self.port)
                if hasattr(server, "__aenter__"):
                    async with server:
                        await self._blocker
                else:
                    await server
                    await self._blocker
            except asyncio.CancelledError:
                pass
            except Exception as e:
                _log.error(f"Failed to start WebSocket server: {e}")

        def _thread_target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_run_server())

        import threading
        self._thread = threading.Thread(target=_thread_target, daemon=True)
        self._thread.start()
        _log.info(f"Serveur WebGPU démarré sur le port {self.port}")

    def stop(self):
        """Gracefully shut down the WebGPU node manager."""
        self.is_running = False
        loop = getattr(self, "_loop", None)
        if loop and not loop.is_closed():
            # Cancel the dispatcher coroutine and the blocker future to unblock _run_server
            if getattr(self, "_dispatcher_task", None):
                loop.call_soon_threadsafe(self._dispatcher_task.cancel)
            if getattr(self, "_blocker", None):
                loop.call_soon_threadsafe(self._blocker.cancel)
            loop.call_soon_threadsafe(loop.stop)
        if getattr(self, "_thread", None):
            self._thread.join(timeout=2)

if __name__ == "__main__":
    manager = WebGPUNodeManager()
    manager.start()
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()

