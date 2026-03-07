import subprocess

try:
    print("Vérification des ports actifs :")
    res = subprocess.run(["netstat", "-tlnp"], capture_output=True, text=True)
    for line in res.stdout.split('\n'):
        if '5000' in line:
            print(line)
except Exception as e:
    print(f"Erreur : {e}")
