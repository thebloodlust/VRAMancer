import re
path = "/home/jeremie/VRAMancer/dashboard/templates/mobile_edge_node.html"
with open(path, "r") as f: content = f.read()
content = re.sub(r"const wsUrl = .*", "const wsUrl = 'ws://192.168.1.21:8560';", content)
with open(path, "w") as f: f.write(content)
print("Fix OK")
