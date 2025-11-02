# tests/test_features_unit.py
import pandas as pd
from src.features import build_features

def test_build_features_basic():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "Open": range(30, 60),
        "High": range(31, 61),
        "Low": range(29, 59),
        "Close": [float(x) for x in range(30, 60)],
        "Volume": [1000 + i for i in range(30)]
    }, index=dates)

    feats = build_features(df)

    # basic checks
    assert "ret_1d" in feats.columns
    assert "vol_5d" in feats.columns
    assert not feats.isna().all().any()
