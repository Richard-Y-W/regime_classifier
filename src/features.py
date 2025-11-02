# src/features.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Convert key numeric columns safely
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute returns & volatility
    df['ret_1d'] = df['Close'].pct_change(1, fill_method=None)
    df['ret_5d'] = df['Close'].pct_change(5, fill_method=None)
    df['ret_21d'] = df['Close'].pct_change(21, fill_method=None)
    df['vol_5d'] = df['ret_1d'].rolling(5).std()
    df['vol_21d'] = df['ret_1d'].rolling(21).std()

    if 'Volume' in df.columns:
        df['vol_z'] = (
            (df['Volume'] - df['Volume'].rolling(21).mean()) /
            df['Volume'].rolling(21).std()
        )

    return df.dropna()


def main(args):
    raw_dir = Path(args.indir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load SPY and VIX
    spy = pd.read_csv(raw_dir / "SPY.csv", index_col=0, parse_dates=True)
    vix = pd.read_csv(raw_dir / "VIX.csv", index_col=0, parse_dates=True)

    # ensure numeric types
    spy = spy.apply(pd.to_numeric, errors="ignore")
    vix = vix.apply(pd.to_numeric, errors="ignore")

    # build SPY features
    spy_feats = build_features(spy)

    # merge VIX Close (since Adj Close no longer exists)
    merged = spy_feats.join(vix['Close'].rename('VIX'), how='left')

    merged.to_csv(out_dir / "features.csv")
    print(f"✅ Saved features to {out_dir / 'features.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ROOT = Path(__file__).resolve().parents[1]
    parser.add_argument("--indir", default=ROOT / "data" / "raw")
    parser.add_argument("--outdir", default=ROOT / "data" / "processed")
    main(parser.parse_args())
