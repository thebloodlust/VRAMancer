# Wrapper Windows vers dashboard réel
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
REAL = os.path.join(ROOT, 'dashboard', 'dashboard_web.py')
if not os.path.exists(REAL):
    print('Dashboard web introuvable:', REAL)
    sys.exit(1)
sys.path.insert(0, ROOT)
runpy.run_path(REAL, run_name='__main__')
