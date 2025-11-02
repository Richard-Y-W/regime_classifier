# conftest.py - ensures pytest can import from src/ on Windows
import sys
from pathlib import Path

# Add the src directory to Python path if not already there
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
