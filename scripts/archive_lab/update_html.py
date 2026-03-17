import re

html_path = "/home/jeremie/VRAMancer/dashboard/templates/mobile_edge_node.html"
with open(html_path, "r") as f:
    html = f.read()

html = re.sub(
    r"const wsUrl = `ws://\${window\.location\.hostname}:8560`;",
    r'const wsUrl = "ws://192.168.1.21:8560";',
    html
)

with open(html_path, "w") as f:
    f.write(html)
print("Updated HTML!")
