with open("core/network/webgpu_node.py", "r") as f:
    text = f.read()

import re
text = re.sub(r'async def _run_server\(\):.*?import threading', 'async def _run_server():\n            self._loop = asyncio.get_running_loop()\n            self._loop.create_task(self._task_dispatcher())\n            async with websockets.serve(self._handler, "0.0.0.0", self.port):\n                import logging\n                logging.info(f"Serveur WebGPU/Mobile lance sur {self.port}")\n                await asyncio.Future()\n                \n        def _thread_target():\n            asyncio.run(_run_server())\n            \n        import threading', text, flags=re.DOTALL)
with open("core/network/webgpu_node.py", "w") as f:
    f.write(text)
