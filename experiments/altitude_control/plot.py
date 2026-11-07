"""Plot a local result directory; use --input data/RUN_NAME."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
if __name__ == '__main__':
    from drone_adaptation.plot import main
    main()
