import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from src.lstm_predictor import LSTMRegime, RegimeDataset, create_sequences
from sklearn.preprocessing import StandardScaler
import argparse

def main(args):
    # ---------------------------------------------------
    # 1️⃣ Load data
    # ---------------------------------------------------
    df = pd.read_csv(args.input, index_col=0, parse_dates=True)

    feature_cols = [
        "ret_1d", "ret_5d", "ret_21d",
        "vol_5d", "vol_21d", "vol_z", "VIX"
    ]
    regime_map = {name: idx for idx, name in enumerate(sorted(df["regime_t5"].unique()))}
    inv_map = {v: k for k, v in regime_map.items()}
    df["regime_t5_id"] = df["regime_t5"].map(regime_map)

    # normalize
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # chronological split
    n = len(df)
    train_end, val_end = int(0.6*n), int(0.8*n)
    test_df = df.iloc[val_end:]

    X_test, y_test = create_sequences(test_df, feature_cols, "regime_t5_id")
    test_loader = DataLoader(RegimeDataset(X_test, y_test), batch_size=32, shuffle=False)

    # ---------------------------------------------------
    # 2️⃣ Load model
    # ---------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegime(
        n_features=len(feature_cols),
        n_hidden=128,
        n_layers=1,
        n_classes=len(regime_map)
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    # ---------------------------------------------------
    # 3️⃣ Evaluate on test set
    # ---------------------------------------------------
    preds, trues = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            out = model(X)
            pred = out.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(y.numpy())

    preds = np.array(preds)
    trues = np.array(trues)

    # ---------------------------------------------------
    # 4️⃣ Metrics
    # ---------------------------------------------------
    print("Classification Report:")
    print(classification_report(trues, preds, target_names=[inv_map[i] for i in sorted(inv_map.keys())]))

    cm = confusion_matrix(trues, preds, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[inv_map[i] for i in sorted(inv_map.keys())])
    disp.plot(cmap="Blues", values_format=".2f")
    plt.title("Normalized Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(args.out_fig)
    plt.show()

    print(f"✅ Confusion matrix figure saved to {args.out_fig}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="models/lstm_best.pt")
    parser.add_argument("--out-fig", default="figs/confusion_matrix.png")
    args = parser.parse_args()
    main(args)
