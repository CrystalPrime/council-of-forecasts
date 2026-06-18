"""
LSTM MCP Server
Multivariate-native time series forecasting with PyTorch.
Supports CSV, TSV, XLSX, XLS input.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
import json
import warnings

from common import load_dataframe

warnings.filterwarnings("ignore")

mcp = FastMCP("lstm-forecast")


def _build_sequences(arr: np.ndarray, lookback: int, target_idx: int):
    X, y = [], []
    for i in range(len(arr) - lookback):
        X.append(arr[i:i + lookback])
        y.append(arr[i + lookback, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


@mcp.tool()
def forecast_lstm(
    file_path: str,
    target_col: str,
    date_col: str = "",
    feature_cols: str = "",
    horizon: int = 12,
    lookback: int = 24,
    epochs: int = 50,
    hidden_size: int = 64,
    num_layers: int = 2,
    sheet_name: str = "",
) -> str:
    """
    Train an LSTM and recursively forecast `horizon` steps.

    Args:
        file_path: Path to CSV, TSV, XLSX, or XLS file.
        target_col: Column to predict.
        date_col: Optional datetime column.
        feature_cols: Comma-separated extra feature columns (multivariate). Empty = univariate.
        horizon: Future steps.
        lookback: Window length used as input.
        epochs: Training epochs.
        hidden_size: LSTM hidden units.
        num_layers: Stacked LSTM layers.
        sheet_name: For Excel files. Empty = first sheet.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return json.dumps({"error": "torch not installed. pip install torch"})

    try:
        df = load_dataframe(file_path, sheet_name=sheet_name if sheet_name != "" else 0)
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)

        features = [c.strip() for c in feature_cols.split(",") if c.strip()]
        cols = [target_col] + [f for f in features if f != target_col]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return json.dumps({"error": f"Missing columns: {missing}"})

        data = df[cols].astype(float).dropna().values
        if len(data) < lookback + 10:
            return json.dumps({
                "error": f"Need at least {lookback + 10} rows after dropna, got {len(data)}"
            })

        # Min-max normalization per column
        mn = data.min(axis=0)
        mx = data.max(axis=0)
        rng = np.where(mx - mn == 0, 1.0, mx - mn)
        scaled = (data - mn) / rng

        target_idx = 0  # target_col is always first
        X, y = _build_sequences(scaled, lookback, target_idx)

        device = "cpu"
        X_t = torch.tensor(X).to(device)
        y_t = torch.tensor(y).to(device)

        class LSTMNet(nn.Module):
            def __init__(self, n_features, hidden, layers):
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        model = LSTMNet(scaled.shape[1], hidden_size, num_layers).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        losses = []
        model.train()
        for ep in range(epochs):
            opt.zero_grad()
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        # Recursive forecast
        model.eval()
        last_window = scaled[-lookback:].copy()
        preds_scaled = []
        with torch.no_grad():
            for _ in range(horizon):
                inp = torch.tensor(last_window[np.newaxis, :, :], dtype=torch.float32)
                p = float(model(inp).item())
                preds_scaled.append(p)
                # Roll window: replace target with prediction, keep other features at last value (naive)
                new_row = last_window[-1].copy()
                new_row[target_idx] = p
                last_window = np.vstack([last_window[1:], new_row])

        preds = np.array(preds_scaled) * rng[target_idx] + mn[target_idx]

        return json.dumps({
            "model": "LSTM",
            "horizon": horizon,
            "lookback": lookback,
            "n_features": int(scaled.shape[1]),
            "features_used": cols,
            "epochs": epochs,
            "final_loss": losses[-1],
            "initial_loss": losses[0],
            "forecast": preds.tolist(),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def describe_lstm() -> str:
    """Return capabilities and when to use LSTM."""
    return json.dumps({
        "name": "LSTM (PyTorch)",
        "best_for": [
            "Long sequences with nonlinear patterns",
            "Multivariate input (multiple features)",
            "Complex temporal dependencies",
            "Large datasets (1000+ points)",
        ],
        "weak_at": ["Small datasets (<200 points)", "Pure linear AR processes", "Interpretability"],
        "needs": ["Sufficient data", "Numeric features (auto-scaled)"],
    })


if __name__ == "__main__":
    mcp.run(transport="stdio")
