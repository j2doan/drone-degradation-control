"""Run this experiment using only the source files in this folder."""
from pathlib import Path
import sys
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    if not any(arg == '--output' or arg.startswith('--output=') for arg in sys.argv[1:]):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        sys.argv += ['--output', str(ROOT / 'data' / ('run-' + stamp + '-' + uuid4().hex[:6]))]
    if not any(arg == '--backend' or arg.startswith('--backend=') for arg in sys.argv[1:]):
        sys.argv += ['--backend', 'mujoco']
    from drone_adaptation.separation.run import main
    main()
