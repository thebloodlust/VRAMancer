from core.network.webgpu_node import WebGPUNodeManager
import asyncio

async def main():
    manager = WebGPUNodeManager()
    await manager._handler(None, None)

