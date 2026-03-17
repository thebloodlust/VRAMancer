with open("core/network/webgpu_node.py", "r") as f:
    text = f.read()

import re
text = text.replace("self._workers_changed = asyncio.Event()", "pass")
text = text.replace("self._workers_changed.set()", "pass")
text = text.replace("await self._workers_changed.wait()", "await asyncio.sleep(1)")
text = text.replace("self._workers_changed.clear()", "pass")
with open("core/network/webgpu_node.py", "w") as f:
    f.write(text)
