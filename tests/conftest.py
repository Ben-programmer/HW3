import sys
from pathlib import Path

# Ensure project root is on sys.path so local packages (data/, features/, models/) can be imported
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
