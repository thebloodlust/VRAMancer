import subprocess
try:
    subprocess.run(["git", "add", "dashboard/dashboard_web.py"], check=True)
    subprocess.run(["git", "commit", "-m", "fix: make swarm animation toggle work globally in JS window object"], check=True)
    print("Fixed script")
except Exception as e:
    pass
