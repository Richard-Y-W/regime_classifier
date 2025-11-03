# src/hmm_regimes.py
"""
Fit GMM/HMM on features, label regimes, save model and produce regimes timeline plot.
Usage (from project root):
python src/hmm_regimes.py --input data/processed/features.csv --out-model models/hmm_model.pkl --out-plot figs/regimes_timeline.png
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Optional import for HMM fallback (if installed)
try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df

def select_features(df: pd.DataFrame):
    # pick numeric features (drop VIX level if you prefer include)
    # You can customize which columns to use; here we select common engineered fields.
    candidate = [c for c in df.columns if any(p in c for p in ("ret_", "vol_", "mom", "VIX", "vol_z"))]
    # Fallback: use all numeric columns if candidate empty
    if not candidate:
        candidate = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[candidate].dropna()

def fit_gmm(X: np.ndarray, n_components: int = 3, random_state: int = 42):
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=random_state)
    gmm.fit(X)
    return gmm

def fit_hmm(X: np.ndarray, n_components: int = 3, random_state: int = 42):
    if not HMM_AVAILABLE:
        raise RuntimeError("hmmlearn not available")
    # hmmlearn expects (n_samples, n_features)
    model = GaussianHMM(n_components=n_components, covariance_type="full", random_state=random_state, n_iter=200)
    model.fit(X)
    return model

def label_clusters(df: pd.DataFrame, cluster_col: str, ret_col: str = "ret_5d", vol_col: str = "vol_21d"):
    """
    Map cluster ids to labels by computing average forward return and vol per cluster.
    Expects df has cluster_col and ret_col (future 5d return) and vol_col.
    Returns mapping dict and a new 'regime' column.
    """
    stats = df.groupby(cluster_col).agg(
        avg_fut_ret=(ret_col, "mean"),
        avg_vol=(vol_col, "mean"),
        count=(cluster_col, "count")
    )
    # create ranking logic:
    # high avg_fut_ret and low avg_vol => Bull
    # negative avg_fut_ret and high avg_vol => Bear
    # remaining with high vol => High-Vol
    # We'll sort clusters by avg_fut_ret ascending to help labels
    stats = stats.reset_index()
    # compute a simple score: ret - vol (normalized)
    stats["score"] = (stats["avg_fut_ret"] - stats["avg_vol"])
    stats = stats.sort_values("score", ascending=False).reset_index(drop=True)

    label_map = {}
    # pick top as Bull, bottom as Bear, middle as High-Vol heuristically
    if len(stats) >= 3:
        label_map[stats.iloc[0][cluster_col]] = "Bull"
        label_map[stats.iloc[-1][cluster_col]] = "Bear"
        for i in range(1, len(stats) - 1):
            label_map[stats.iloc[i][cluster_col]] = "High-Vol"
    elif len(stats) == 2:
        label_map[stats.iloc[0][cluster_col]] = "Bull"
        label_map[stats.iloc[1][cluster_col]] = "Bear"
    elif len(stats) == 1:
        label_map[stats.iloc[0][cluster_col]] = "Bull"
    else:
        raise ValueError("No clusters found to label.")

    df["regime"] = df[cluster_col].map(label_map)
    return label_map, df

def plot_regimes(df: pd.DataFrame, price_col: str = "Adj Close", out_path: Path = None):
    """
    Plot price with colored regime bands underneath.
    Expects df index is datetime and has 'regime' column.
    """
    plt.figure(figsize=(14, 5))
    ax = plt.gca()
    # Price
    if price_col not in df.columns:
        # try to locate a close-like column
        cand = [c for c in df.columns if "Adj" in c or "Close" in c]
        if cand:
            price_col = cand[0]
        else:
            raise ValueError("No price column found for plotting")

    df[price_col].plot(ax=ax, label="Price", linewidth=1.2)

    # Map regimes to colors
    color_map = {"Bull": "#2ca02c", "Bear": "#d62728", "High-Vol": "#ff7f0e"}
    # If there are other labels, assign colors cyclically
    unique_regs = df["regime"].dropna().unique().tolist()
    # build region spans
    prev_reg = None
    start = None
    for i, (ts, row) in enumerate(df.iterrows()):
        reg = row["regime"]
        if i == 0:
            prev_reg = reg
            start = ts
            continue
        if reg != prev_reg:
            # draw band from start to previous timestamp
            ax.axvspan(start, ts, ymin=0.02, ymax=0.2, facecolor=color_map.get(prev_reg, "#999999"), alpha=0.25)
            start = ts
            prev_reg = reg
    # final span
    if start is not None and prev_reg is not None:
        ax.axvspan(start, df.index[-1], ymin=0.02, ymax=0.2, facecolor=color_map.get(prev_reg, "#999999"), alpha=0.25)

    ax.set_title("Regimes timeline")
    ax.legend()
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved timeline plot to {out_path}")
    else:
        plt.show()
    plt.close()

def main(args):
    inp = Path(args.input)
    out_model = Path(args.out_model)
    out_plot = Path(args.out_plot)

    # 1) load
    df = load_features(inp)

    # 2) ensure we have forward returns for labelling:
    # compute 5-day forward returns if not present
    if "ret_5d" not in df.columns:
        df["ret_5d"] = df["Adj Close"].pct_change(5).shift(-5)

    # keep index aligned, drop rows with NaNs after computing features
    df = df.dropna()

    # 3) select columns for clustering
    feats_df = select_features(df)
    feats_df = feats_df.dropna()
    if feats_df.shape[0] < 50:
        raise RuntimeError("Not enough rows in features for clustering. Need more data.")

    # 4) scale
    scaler = StandardScaler()
    X = scaler.fit_transform(feats_df.values)

    # optional PCA denoising
    if args.pca and args.pca < X.shape[1]:
        pca = PCA(n_components=args.pca, random_state=42)
        X_red = pca.fit_transform(X)
        X_used = X_red
    else:
        X_used = X

    # 5) try GMM first
    gmm = fit_gmm(X_used, n_components=args.n_components)
    cluster_ids = gmm.predict(X_used)
    # attach back to the original df (align by index)
    feats_df = feats_df.assign(cluster=cluster_ids)
    # join cluster into df
    df = df.join(feats_df["cluster"], how="left")

    # 6) label clusters
    mapping, df_labeled = label_clusters(df, "cluster", ret_col="ret_5d", vol_col="vol_21d")

    # 7) persist scaler and model
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": gmm, "mapping": mapping}, out_model)
    joblib.dump(scaler, out_model.parent / "scaler.pkl")
    print(f"Saved model to {out_model} and scaler to {out_model.parent / 'scaler.pkl'}")

    # 8) plot timeline
    plot_regimes(df_labeled, price_col="Adj Close", out_path=out_plot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit GMM/HMM to discover market regimes")
    parser.add_argument("--input", type=str, default="data/processed/features.csv", help="Processed features CSV")
    parser.add_argument("--out-model", type=str, default="models/hmm_model.pkl", help="Where to save model (joblib)")
    parser.add_argument("--out-plot", type=str, default="figs/regimes_timeline.png", help="Timeline plot")
    parser.add_argument("--n-components", type=int, default=3, help="Number of regimes")
    parser.add_argument("--pca", type=int, default=0, help="Optional PCA components (0 to disable)")
    args = parser.parse_args()
    main(args)
