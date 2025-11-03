# tests/test_hmm_regimes_basic.py
import os
from pathlib import Path
import joblib
import pandas as pd


def test_hmm_regimes_runs_and_saves(tmp_path, monkeypatch):
    # Use the small processed features CSV if available; otherwise skip.
    root = Path.cwd()
    feat = root / "data" / "processed" / "features.csv"
    assert feat.exists(), "features.csv not found; run src/features.py first"

    out_model = root / "models" / "hmm_test_model.pkl"
    out_plot = tmp_path / "regimes.png"

    # run the script directly
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "src/hmm_regimes.py",
                       "--input", str(feat),
                       "--out-model", str(out_model),
                       "--out-plot", str(out_plot),
                       "--n-components", "3",
                       "--pca", "0"])


    assert out_model.exists(), "Model file was not created"
    # load and check expected keys
    data = joblib.load(out_model)
    assert "model" in data and "mapping" in data
