# core/network/remote_executor.py

import socket
import pickle

class RemoteBlock:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port

    def forward(self, x):
        """
        Envoie le tensor *x* à une machine distante et reçoit le résultat.
        ⚠️ Nécessite un serveur distant en écoute.
        """
        data = pickle.dumps(x.cpu())
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.ip, self.port))
            s.sendall(data)
            result = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                result += chunk
        return pickle.loads(result)
