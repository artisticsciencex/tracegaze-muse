import sys
from pathlib import Path

# make `neon_trim` importable when running pytest from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
