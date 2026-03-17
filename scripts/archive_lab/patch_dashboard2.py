with open("dashboard/dashboard_web.py", "r") as f:
    text = f.read()

route = """
@app.route("/node")
def api_mobile_node():
    import os
    path = os.path.join(os.path.dirname(__file__), "templates", "mobile_edge_node.html")
    with open(path, "r") as f:
        return f.read()
"""
if "/node" not in text:
    text = text.replace("def launch():", route + "\ndef launch():")
    with open("dashboard/dashboard_web.py", "w") as f:
        f.write(text)
    print("Injected via launch()")
