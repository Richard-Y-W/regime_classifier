# tests/test_integration_datafiles.py
from pathlib import Path

def test_raw_and_processed_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "data" / "raw" / "SPY.csv").exists(), "SPY.csv missing in data/raw"
    assert (root / "data" / "raw" / "VIX.csv").exists(), "VIX.csv missing in data/raw"
    assert (root / "data" / "processed" / "features.csv").exists(), "features.csv missing in data/processed"
