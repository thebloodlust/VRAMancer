import subprocess
import sys

def main():
    print("Tentative d'installation de Flask avec l'interpreteur global...")
    try:
        subprocess.run(["/usr/local/python/3.12.1/bin/python3", "-m", "pip", "install", "flask", "flask-socketio", "websockets", "pynvml", "prometheus_client"], check=True)
        print("Succès !")
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == '__main__':
    main()
