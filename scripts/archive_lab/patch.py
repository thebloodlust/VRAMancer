import re

with open("core/network/webgpu_node.py", "r") as f:
    content = f.read()

# Etape 1: on supprime la section problématique à la fin de _handler
start_str = "            # Trouver un client libre"
end_str = "                await asyncio.sleep(0.01)\n"

start_idx = content.find(start_str)
end_idx = content.find(end_str) + len(end_str)

if start_idx != -1 and end_idx != -1 and "await self.task_queue.get()" in content[start_idx:end_idx]:
    content = content[:start_idx] + content[end_idx:]

# Etape 2: on ajoute _task_dispatcher avant submit_tensor
dispatcher = """
    async def _task_dispatcher(self):
        \"\"\"Smart Task Dispatcher with Load-Balancing & GPU ranking.\"\"\"
        while True:
            # 1. Attends qu'une tâche soit disponible
            task = await self.task_queue.get()
            
            # 2. Cherche un client libre
            while True:
                available_clients = [(cid, c) for cid, c in self.clients.items() if not c.get("busy", False)]
                if not available_clients:
                    await asyncio.sleep(0.01)
                    continue
                
                # Smart Load Balancing: prioriser les meilleurs GPUs (M-series, RTX, etc)
                def get_gpu_score(client_data):
                    gpu_name = client_data.get("gpu", "").lower()
                    if "rtx" in gpu_name or "m3 max" in gpu_name or "m4 max" in gpu_name: return 100
                    if "m2" in gpu_name or "m1 max" in gpu_name or "rx 7900" in gpu_name or "m3 pro" in gpu_name: return 80
                    if "m1" in gpu_name or "gtx 1080" in gpu_name: return 50
                    return 10 # generic
                
                # Trie par score décroissant (meilleur GPU d'abord)
                available_clients.sort(key=lambda x: get_gpu_score(x[1]), reverse=True)
                
                best_client_id = available_clients[0][0]
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
"""

if "def _task_dispatcher" not in content:
    content = content.replace("    def submit_tensor(", dispatcher + "\n    def submit_tensor(")

with open("core/network/webgpu_node.py", "w") as f:
    f.write(content)

print("Patched successfully")
