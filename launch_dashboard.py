import subprocess
import os

# Launch in background
env = os.environ.copy()
env['VRM_PRODUCTION'] = '0'
subprocess.Popen(["python3", "dashboard/launcher.py", "--mode", "web"], env=env)
print("Dashboard Web launched on port 5000")
