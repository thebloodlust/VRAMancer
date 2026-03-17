import re

with open("core/network/webgpu_node.py", "r") as f:
    content = f.read()

# Etape 1: Ajouter l'except manquant
except_block = """        except ConnectionClosed:
            _log.info(f"Client {client_id} déconnecté proprement.")
        except Exception as e:
            _log.error(f"Erreur client {client_id}: {e}")
        finally:
            await self._disconnect_client(client_id)
            
"""
content = content.replace("    async def _task_dispatcher(self):", except_block + "    async def _task_dispatcher(self):")

with open("core/network/webgpu_node.py", "w") as f:
    f.write(content)

