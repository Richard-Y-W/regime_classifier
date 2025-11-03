import pandas as pd
from pathlib import Path
import argparse

def make_supervised(df: pd.DataFrame, target_col: str = "regime", horizon: int = 5):
    """
    Shift the regime column to create a T+horizon target.
    Returns a DataFrame with new column 'regime_t{horizon}'.
    """
    df = df.copy()
    df[f"{target_col}_t{horizon}"] = df[target_col].shift(-horizon)
    df = df.dropna(subset=[f"{target_col}_t{horizon}"])
    return df

def main(args):
    input_path = Path(args.input)
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    print(f"Loaded {df.shape[0]} rows from {input_path}")

    df_supervised = make_supervised(df, target_col="regime", horizon=args.horizon)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_supervised.to_csv(out_path)
    print(f"✅ Saved supervised dataset to {out_path}")
    print(f"Columns: {df_supervised.columns.tolist()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()
    main(args)
