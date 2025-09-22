import argparse
from dashboard.flask_app import app

def main():
    parser = argparse.ArgumentParser(description="Lancer le dashboard VRAMancer")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"📊 Dashboard lancé sur http://localhost:{args.port}")
    app.run(port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
