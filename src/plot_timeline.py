import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.lstm_predictor import LSTMRegime, create_sequences, RegimeDataset
from torch.utils.data import DataLoader
import seaborn as sns

# Setup
ROOT = Path.cwd()
data_path = ROOT / "data" / "processed" / "supervised_t5.csv"
model_path = ROOT / "models" / "lstm_best.pt"

df = pd.read_csv(data_path, index_col=0, parse_dates=True)

feature_cols = ["ret_1d", "ret_5d", "ret_21d", "vol_5d", "vol_21d", "vol_z", "VIX"]
regime_map = {name: idx for idx, name in enumerate(sorted(df["regime_t5"].unique()))}
inv_map = {v: k for k, v in regime_map.items()}
df["regime_t5_id"] = df["regime_t5"].map(regime_map)

# Scale features
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Split chronologically
n = len(df)
train_end, val_end = int(0.6*n), int(0.8*n)
test_df = df.iloc[val_end:].copy()

X_test, y_test = create_sequences(test_df, feature_cols, "regime_t5_id")
test_loader = DataLoader(RegimeDataset(X_test, y_test), batch_size=32, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRegime(
    n_features=len(feature_cols),
    n_hidden=128,
    n_layers=1,
    n_classes=len(regime_map)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Predict
preds = []
with torch.no_grad():
    for X, _ in test_loader:
        X = X.to(device)
        out = model(X)
        preds.extend(out.argmax(dim=1).cpu().numpy())

# Align predictions with test_df index
df_test_trimmed = test_df.iloc[-len(preds):].copy()
df_test_trimmed["predicted_regime_id"] = preds
df_test_trimmed["predicted_regime"] = df_test_trimmed["predicted_regime_id"].map(inv_map)

# Plot
sns.set(style="whitegrid", context="talk")
plt.figure(figsize=(12, 6))

plt.plot(df_test_trimmed.index, df_test_trimmed["Close"], color="black", lw=1.5, label="SPY Close")

# Overlay true and predicted regimes
colors = {"Bull": "#4CAF50", "Bear": "#F44336", "High-Vol": "#FFC107"}
plt.fill_between(df_test_trimmed.index, df_test_trimmed["Close"].min(),
                 df_test_trimmed["Close"].max(),
                 where=df_test_trimmed["regime_t5"]=="Bull", color=colors["Bull"], alpha=0.1)
plt.fill_between(df_test_trimmed.index, df_test_trimmed["Close"].min(),
                 df_test_trimmed["Close"].max(),
                 where=df_test_trimmed["regime_t5"]=="Bear", color=colors["Bear"], alpha=0.1)
plt.fill_between(df_test_trimmed.index, df_test_trimmed["Close"].min(),
                 df_test_trimmed["Close"].max(),
                 where=df_test_trimmed["regime_t5"]=="High-Vol", color=colors["High-Vol"], alpha=0.1)

plt.title("Predicted vs. True Regimes (LSTM Test Set)")
plt.xlabel("Date")
plt.ylabel("SPY Close")
plt.legend()
plt.tight_layout()
plt.savefig("figs/lstm_regime_timeline.png")
plt.show()

print("✅ Saved timeline to figs/lstm_regime_timeline.png")
