import subprocess

try:
    subprocess.run(["git", "add", "dashboard/dashboard_web.py"], check=True)
    subprocess.run(["git", "commit", "-m", "feat: Add Swarm Attention Organic Neural Network dashboard visualizer"], check=True)
    print("Dashboard visualizer committed successfully")
except subprocess.CalledProcessError as e:
    print(f"Failed: {e}")
