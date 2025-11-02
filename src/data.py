# src/data.py
import argparse
import yfinance as yf
import pandas as pd
from pathlib import Path

def download_ticker(ticker: str, start: str = "2010-01-01", end: str = None):
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.index = pd.to_datetime(df.index)
    return df

def main(args):
    # --- NEW: make path relative to project root ---
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in args.tickers:
        df = download_ticker(t, start=args.start, end=args.end)
        out_path = out_dir / f"{t}.csv"
        df.to_csv(out_path)
        print(f"✅ Saved {t} to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["SPY", "VIX"])
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default="data/raw")
    main(parser.parse_args())
