import re

with open('core/network/webgpu_node.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Completely rewrite the handler to be absolute barebones and never reject anyone
new_text = re.sub(
    r'    async def _handler\(self, websocket, path\):.*?self\.clients\[client_id\] = \{',
    r'''    async def _handler(self, websocket, path=""):
        client_id = uuid.uuid4().hex[:8]
        _log.info(f"Nouvelle connexion WebGPU: {client_id}")
        
        self.clients[client_id] = {''',
    text,
    flags=re.DOTALL
)

with open('core/network/webgpu_node.py', 'w', encoding='utf-8') as f:
    f.write(new_text)

