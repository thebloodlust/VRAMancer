"""Rejoindre un nœud existant en une commande, quel que soit l'OS.

Le nœud principal lance `vramancer invite` : un petit service HTTP (bibliothèque
standard, séparé de l'API) qui affiche deux commandes à copier sur l'autre machine :

    Linux / macOS : curl -fsSL 'http://<nœud>:5055/join.sh?t=<jeton>' | sh
    Windows       : irm 'http://<nœud>:5055/join.ps1?t=<jeton>' | iex

Le script servi détecte l'OS, l'architecture et la présence de Vulkan, télécharge le
`rpc-server` précompilé de llama.cpp de la MÊME version que le nœud principal (le
protocole RPC change d'une version à l'autre), le lance, puis s'enregistre auprès du
nœud. Rien à installer : ni Python, ni CUDA, ni ROCm (Vulkan couvre NVIDIA, AMD et
Intel ; Metal sur Mac). `vramancer serve` utilise ensuite les nœuds enregistrés
et joignables.

Sécurité — à lire : `rpc-server` n'a AUCUNE authentification. Quiconque atteint son
port peut y exécuter des calculs et lire la mémoire qu'on lui confie. Le script le lie
donc à l'adresse que le nœud principal a vue (celle du réseau local ou du tunnel),
jamais à 0.0.0.0, et le jeton empêche seulement l'enregistrement par un inconnu.
Réseau local ou tunnel (WireGuard, Tailscale) uniquement.
"""
from __future__ import annotations

import hmac
import json
import logging
import re
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, List, Optional
from urllib.parse import parse_qs, urlparse

log = logging.getLogger(__name__)

NODES_FILE = Path.home() / ".cache" / "vramancer" / "nodes.json"
RPC_PORT = 50052
_LOCK = threading.Lock()

# Asset llama.cpp par (OS, architecture) — noms vérifiés sur la release b11112.
# Le build Vulkan contient aussi le CPU : sans GPU, rpc-server sert la RAM + le CPU.
_SH = r"""#!/bin/sh
# VRAMancer — rejoindre le nœud {coordinator} (généré par `vramancer invite`)
set -e
TAG="{tag}"; BIND="{bind}"; PORT="${{VRM_RPC_PORT:-{port}}}"
REL="https://github.com/ggml-org/llama.cpp/releases/download/$TAG"
OS=$(uname -s); ARCH=$(uname -m)
VK=0
if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q libvulkan.so.1; then VK=1; fi
if [ "$VK" = 0 ] && ls /usr/lib*/libvulkan.so.1 /usr/lib/*/libvulkan.so.1 >/dev/null 2>&1; then VK=1; fi
case "$OS-$ARCH" in
  Linux-x86_64)  if [ "$VK" = 1 ]; then A=ubuntu-vulkan-x64; else A=ubuntu-x64; fi ;;
  Linux-aarch64|Linux-arm64) if [ "$VK" = 1 ]; then A=ubuntu-vulkan-arm64; else A=ubuntu-arm64; fi ;;
  Darwin-arm64)  A=macos-arm64 ;;
  Darwin-x86_64) A=macos-x64 ;;
  *) echo "Plateforme non prise en charge : $OS-$ARCH" >&2; exit 1 ;;
esac
DIR="${{VRM_NODE_DIR:-$HOME/.local/share/vramancer-node}}/$TAG-$A"
if [ ! -d "$DIR" ]; then
  echo "Téléchargement de llama.cpp $TAG ($A)…"
  mkdir -p "$DIR"
  curl -fsSL "$REL/llama-$TAG-bin-$A.tar.gz" | tar xz -C "$DIR"
fi
BIN=$(find "$DIR" -type f \( -name ggml-rpc-server -o -name rpc-server -o -name llama-rpc-server \) | head -n 1)
[ -n "$BIN" ] || {{ echo "rpc-server introuvable dans l'archive" >&2; exit 1; }}
chmod +x "$BIN"
LIB=$(dirname "$BIN")
DEV=""; [ -n "$VRM_RPC_DEVICE" ] && DEV="-d $VRM_RPC_DEVICE"
echo "Lancement de rpc-server sur $BIND:$PORT (réseau local / tunnel uniquement : pas d'authentification)"
LD_LIBRARY_PATH="$LIB" DYLD_LIBRARY_PATH="$LIB" nohup "$BIN" -H "$BIND" -p "$PORT" -c $DEV \
  > "$DIR/rpc-server.log" 2>&1 &
sleep 3
kill -0 $! 2>/dev/null || {{ echo "rpc-server n'a pas démarré :" >&2; tail -5 "$DIR/rpc-server.log" >&2; exit 1; }}
curl -fsS -X POST "{coordinator}/register" -H "Content-Type: application/json" \
  -d "{{\"token\":\"{token}\",\"port\":$PORT,\"os\":\"$OS-$ARCH\",\"asset\":\"$A\"}}"
echo
echo "Nœud ajouté. Journal : $DIR/rpc-server.log — arrêt : kill $!"
"""

_PS1 = r"""# VRAMancer — rejoindre le nœud {coordinator} (généré par `vramancer invite`)
$ErrorActionPreference = "Stop"
$Tag = "{tag}"; $Bind = "{bind}"; $Port = if ($env:VRM_RPC_PORT) {{ $env:VRM_RPC_PORT }} else {{ {port} }}
$A = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") {{ "win-cpu-arm64" }} else {{ "win-vulkan-x64" }}
$Dir = Join-Path $env:LOCALAPPDATA "vramancer-node\$Tag-$A"
if (-not (Test-Path $Dir)) {{
  Write-Host "Téléchargement de llama.cpp $Tag ($A)…"
  New-Item -ItemType Directory -Force -Path $Dir | Out-Null
  $Zip = Join-Path $env:TEMP "llama-$Tag-$A.zip"
  Invoke-WebRequest "https://github.com/ggml-org/llama.cpp/releases/download/$Tag/llama-$Tag-bin-$A.zip" -OutFile $Zip
  Expand-Archive $Zip -DestinationPath $Dir -Force
}}
$Bin = Get-ChildItem $Dir -Recurse -Include ggml-rpc-server.exe,rpc-server.exe,llama-rpc-server.exe | Select-Object -First 1
if (-not $Bin) {{ throw "rpc-server introuvable dans l'archive" }}
$Args = @("-H", $Bind, "-p", "$Port", "-c")
if ($env:VRM_RPC_DEVICE) {{ $Args += @("-d", $env:VRM_RPC_DEVICE) }}
Write-Host "Lancement de rpc-server sur ${{Bind}}:$Port (réseau local / tunnel uniquement : pas d'authentification)"
$P = Start-Process -FilePath $Bin.FullName -ArgumentList $Args -WorkingDirectory $Bin.DirectoryName -WindowStyle Hidden -PassThru
Start-Sleep 3
if ($P.HasExited) {{ throw "rpc-server n'a pas démarré" }}
$Body = @{{ token = "{token}"; port = [int]$Port; os = "Windows-$env:PROCESSOR_ARCHITECTURE"; asset = $A }} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "{coordinator}/register" -ContentType "application/json" -Body $Body
Write-Host "Nœud ajouté. Arrêt : Stop-Process -Id $($P.Id)"
"""


def render_script(kind: str, coordinator: str, token: str, tag: str, bind: str,
                  port: int = RPC_PORT) -> str:
    """Script d'amorçage (`sh` ou `ps1`) pour la machine qui rejoint."""
    for name, v in (("tag", tag), ("bind", bind), ("token", token)):
        if not re.fullmatch(r"[A-Za-z0-9._:\-]+", v):          # injecté dans un script
            raise ValueError(f"{name} invalide : {v!r}")
    tpl = _PS1 if kind == "ps1" else _SH
    return tpl.format(coordinator=coordinator.rstrip("/"), token=token, tag=tag,
                      bind=bind, port=int(port))


def binary_tag(binary) -> Optional[str]:
    """Version llama.cpp (bNNNNN) du binaire local, lue dans son chemin."""
    m = re.search(r"\b(b\d{4,6})\b", str(binary).replace("-", " ").replace("/", " "))
    return m.group(1) if m else None


# ── Registre des nœuds ──────────────────────────────────────────────────────

def _load() -> dict:
    try:
        return json.loads(NODES_FILE.read_text())
    except Exception:
        return {}


def register_node(host: str, port: int, info: Optional[dict] = None) -> str:
    key = f"{host}:{int(port)}"
    with _LOCK:
        nodes = _load()
        nodes[key] = dict(info or {}, host=host, port=int(port), joined_at=time.time())
        NODES_FILE.parent.mkdir(parents=True, exist_ok=True)
        NODES_FILE.write_text(json.dumps(nodes, indent=2))
    return key


def _reachable(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def joined_rpc_hosts(timeout: float = 1.0) -> List[str]:
    """Nœuds enregistrés ET joignables maintenant (un nœud éteint est ignoré, pas fatal)."""
    return [k for k, n in _load().items() if _reachable(n["host"], n["port"], timeout)]


# ── Service d'invitation ────────────────────────────────────────────────────

def make_handler(token: str, tag: str, public_url: str,
                 on_join: Optional[Callable[[str, dict], None]] = None,
                 check_reachable: bool = True):
    class Handler(BaseHTTPRequestHandler):
        server_version = "vramancer-invite"

        def log_message(self, fmt, *args):
            log.debug("invite: " + fmt, *args)

        def _send(self, code: int, body: str, ctype: str = "text/plain; charset=utf-8"):
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _token_ok(self, given: Optional[str]) -> bool:
            return bool(given) and hmac.compare_digest(str(given), token)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path not in ("/join.sh", "/join.ps1"):
                return self._send(404, "introuvable\n")
            if not self._token_ok(parse_qs(u.query).get("t", [None])[0]):
                return self._send(403, "jeton invalide\n")
            # Adresse de la machine telle que le nœud la voit : c'est là que rpc-server
            # doit écouter (réseau local ou tunnel), jamais 0.0.0.0.
            bind = self.client_address[0]
            kind = "ps1" if u.path.endswith(".ps1") else "sh"
            self._send(200, render_script(kind, public_url, token, tag, bind))

        def do_POST(self):
            if urlparse(self.path).path != "/register":
                return self._send(404, "introuvable\n")
            try:
                n = min(int(self.headers.get("Content-Length", "0")), 4096)
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return self._send(400, "JSON invalide\n")
            if not self._token_ok(body.get("token")):
                return self._send(403, "jeton invalide\n")
            host, port = self.client_address[0], int(body.get("port", RPC_PORT))
            if check_reachable and not _reachable(host, port, 3.0):
                return self._send(502, f"rpc-server injoignable sur {host}:{port}\n")
            info = {k: str(body[k])[:64] for k in ("os", "asset") if k in body}
            key = register_node(host, port, info)
            if on_join:
                on_join(key, info)
            self._send(200, json.dumps({"ok": True, "node": key}), "application/json")

    return Handler


def serve_invites(tag: str, host: str = "0.0.0.0", port: int = 5055,
                  public_host: Optional[str] = None, token: Optional[str] = None,
                  on_join: Optional[Callable[[str, dict], None]] = None):
    """Démarre le service d'invitation ; renvoie (serveur, jeton, url publique)."""
    token = token or secrets.token_urlsafe(16).replace("_", "x").replace("-", "y")
    public_host = public_host or _guess_lan_ip()
    url = f"http://{public_host}:{port}"
    httpd = ThreadingHTTPServer((host, port), make_handler(token, tag, url, on_join))
    return httpd, token, url


def _guess_lan_ip() -> str:
    """Adresse locale de la route par défaut (aucun paquet n'est envoyé)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


__all__ = ["render_script", "binary_tag", "register_node", "joined_rpc_hosts",
           "serve_invites", "make_handler", "NODES_FILE"]
