import sys
from pathlib import Path

# Add src to import path for tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
