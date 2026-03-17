import re

with open("dashboard/dashboard_web.py", "r") as f:
    text = f.read()

log_hook = """

import logging
from collections import deque
import time

# Ring buffer in memory to keep the last 200 logs for Supervision
log_buffer = deque(maxlen=200)

class WebsocketLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            log_entry = {
                "time": time.time(),
                "level": record.levelname,
                "module": record.module,
                "msg": msg
            }
            log_buffer.append(log_entry)
            # if socketio is present, broadcast it
            if globals().get('socketio'):
                socketio.emit('new_log', log_entry)
        except Exception:
            pass

try:
    global_logger = logging.getLogger("vramancer")
    ws_handler = WebsocketLogHandler()
    ws_handler.setFormatter(logging.Formatter('%(message)s'))
    global_logger.addHandler(ws_handler)
except Exception:
    pass


@app.route("/api/debug/logs")
def api_debug_logs():
    return jsonify(list(log_buffer))

"""

if "WebsocketLogHandler" not in text:
    text = text.replace("app = Flask(__name__)", "app = Flask(__name__)\n" + log_hook)

with open("dashboard/dashboard_web.py", "w") as f:
    f.write(text)

print("Logs hook injected. Backend API /api/debug/logs now exists.")
