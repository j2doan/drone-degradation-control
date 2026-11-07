"""Run this experiment's local flight script."""
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if __name__ == '__main__':
    runpy.run_path(str(ROOT / '03_collect_data.py'), run_name='__main__')
