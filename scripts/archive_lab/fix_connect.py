path = "/home/jeremie/VRAMancer/dashboard/templates/mobile_edge_node.html"
with open(path, "r") as f: content = f.read()

import re
content = re.sub(r'let wsUrl =.*', 'let wsUrl = "ws://192.168.1.21:8560/";', content)
content = re.sub(r'const wsUrl =.*', 'let wsUrl = "ws://192.168.1.21:8560/";', content)

with open(path, "w") as f: f.write(content)
print("Fix syntax error OK")
