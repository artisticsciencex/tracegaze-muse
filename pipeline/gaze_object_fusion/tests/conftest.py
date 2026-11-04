import sys
from pathlib import Path

# make pipeline packages importable when running pytest from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
