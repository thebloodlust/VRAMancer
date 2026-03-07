import subprocess

try:
    subprocess.run(["git", "add", "core/paged_attention.py", "core/network/webgpu_node.py"], check=True)
    subprocess.run(["git", "commit", "-m", "feat: Introduce Swarm Attention distributed KV cache via WebGPU and PagedAttention VTP offload."], check=True)
    print("Committed successfully")
except subprocess.CalledProcessError as e:
    print(f"Failed: {e}")
