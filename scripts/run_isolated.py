"""Run packet verification in a fresh user/network namespace, never host networking."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
env = dict(os.environ, CYBERTGN_HOST_NET_NS=os.readlink('/proc/self/ns/net'))
result = subprocess.run(['unshare', '--user', '--map-root-user', '--net', str(root / 'venv/bin/python'),
                         str(root / 'scripts/verify_live.py')], env=env, cwd=root)
sys.exit(result.returncode)
