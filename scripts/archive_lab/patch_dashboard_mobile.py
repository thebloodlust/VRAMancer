import re

with open("dashboard/dashboard_web.py", "r") as f:
    text = f.read()

# Make sure we don't duplicate
if 'def api_mobile_node():' not in text:
    route_injection = """
@app.route("/node")
def api_mobile_node():
    import os
    path = os.path.join(os.path.dirname(__file__), "templates", "mobile_edge_node.html")
    with open(path, "r") as f:
        return f.read()
"""
    # Simply append before the main loop or at the very end before if __name__ == '__main__'
    text = text.replace('if __name__ == "__main__":', route_injection + '\n\nif __name__ == "__main__":')
    
    with open("dashboard/dashboard_web.py", "w") as f:
        f.write(text)
    
    print("Dashboard route /node added!")
else:
    print("Dashboard route /node already exists.")
