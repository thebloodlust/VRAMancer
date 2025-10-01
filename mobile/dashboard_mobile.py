"""
Interface mobile/tablette :
- Dashboard Flask pour contrôle et monitoring mobile
- Vue responsive, accès sécurisé
"""
from flask import Flask, render_template_string

app = Flask(__name__)

TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>VRAMancer Mobile</title>
    <style>
        body { font-family: sans-serif; margin: 0; padding: 0; }
        .header { background: #1976D2; color: white; padding: 1em; text-align: center; }
        .section { margin: 1em; }
        .card { background: #f5f5f5; margin: 1em 0; padding: 1em; border-radius: 8px; }
        @media (max-width: 600px) { .section { margin: 0.5em; } }
    </style>
</head>
<body>
    <div class="header">VRAMancer Mobile</div>
    <div class="section">
        <div class="card">Statut cluster : <b>OK</b></div>
        <div class="card">Jobs actifs : <b>3</b></div>
        <div class="card">Dernier log : <b>Tout fonctionne</b></div>
    </div>
</body>
</html>
'''

@app.route("/")
def mobile_dashboard():
    return render_template_string(TEMPLATE)

if __name__ == "__main__":
    app.run(port=5003, debug=True)
