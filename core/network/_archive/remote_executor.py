# core/network/remote_executor.py

import socket
import io
import torch

class RemoteBlock:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port

    def forward(self, x):
        """
        Envoie le tensor *x* à une machine distante et reçoit le résultat.
         Nécessite un serveur distant en écoute.
        """
        buffer = io.BytesIO()
        torch.save(x.cpu(), buffer)
        data = buffer.getvalue()
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.ip, self.port))
            s.sendall(data)
            result = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                result += chunk
                
        return torch.load(io.BytesIO(result), weights_only=True)
