"""
API “No Code” :
- Interface drag & drop pour pipelines IA
- Endpoints, exemple pipeline
"""
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/no-code/pipeline", methods=["POST"])
def create_pipeline():
    pipeline = request.json
    print(f"[NoCode] Pipeline reçu : {pipeline}")
    return jsonify({"ok": True, "pipeline": pipeline})

if __name__ == "__main__":
    app.run(port=5004, debug=True)
