import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==============================
# Dataset
# ==============================
class RegimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==============================
# Sequence builder
# ==============================
def create_sequences(df, feature_cols, target_col, seq_len=21):
    X, y = [], []
    for i in range(len(df) - seq_len):
        seq_x = df[feature_cols].iloc[i:i+seq_len].values
        seq_y = df[target_col].iloc[i+seq_len]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# ==============================
# LSTM model
# ==============================
class LSTMRegime(nn.Module):
    def __init__(self, n_features, n_hidden=128, n_layers=1, n_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, n_hidden, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]       # last time step
        out = self.fc(out)
        return out

# ==============================
# Main training script
# ==============================
def main(args):
    df = pd.read_csv(args.input, index_col=0, parse_dates=True)
    feature_cols = [
        "ret_1d", "ret_5d", "ret_21d",
        "vol_5d", "vol_21d", "vol_z", "VIX"
    ]

    # encode regimes numerically
    regime_map = {name: idx for idx, name in enumerate(sorted(df["regime_t5"].unique()))}
    df["regime_t5_id"] = df["regime_t5"].map(regime_map)

    # normalize features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # split chronologically (60/20/20)
    n = len(df)
    train_end, val_end = int(0.6*n), int(0.8*n)
    train_df, val_df, test_df = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    # make sequences
    X_train, y_train = create_sequences(train_df, feature_cols, "regime_t5_id")
    X_val, y_val = create_sequences(val_df, feature_cols, "regime_t5_id")
    X_test, y_test = create_sequences(test_df, feature_cols, "regime_t5_id")

    train_loader = DataLoader(RegimeDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(RegimeDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(RegimeDataset(X_test, y_test), batch_size=32, shuffle=False)

    # model + training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegime(n_features=len(feature_cols), n_hidden=128, n_layers=1, n_classes=len(regime_map))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    for epoch in range(30):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                val_loss += criterion(out, y).item()

        print(f"Epoch {epoch+1:02d} | Train {train_loss/len(train_loader):.4f} | Val {val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.out_model)
            print(f"✅ Saved new best model to {args.out_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-model", default="models/lstm_best.pt")
    args = parser.parse_args()
    main(args)
