"""
Prophet MCP Server
Univariate forecast with optional extra regressors (multivariate).
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

mcp = FastMCP("prophet-forecast")


@mcp.tool()
def forecast_prophet(
    file_path: str,
    target_col: str,
    date_col: str,
    horizon: int = 12,
    freq: str = "MS",
    regressor_cols: str = "",
    yearly: bool = True,
    weekly: bool = False,
    daily: bool = False,
    sheet_name: str = "",
) -> str:
    """
    Fit Prophet and forecast `horizon` future periods.

    Args:
        file_path: Path to CSV, TSV, XLSX, or XLS file.
        target_col: Column to forecast (becomes 'y').
        date_col: Datetime column (becomes 'ds'). REQUIRED for Prophet.
        horizon: Future periods.
        freq: Pandas offset alias ('D','MS','W','H', etc.).
        regressor_cols: Comma-separated extra regressors (multivariate).
        yearly/weekly/daily: Seasonality toggles.
        sheet_name: For Excel files. Empty = first sheet.

    Returns:
        JSON with forecast mean, lower/upper bounds, changepoints.
    """
    try:
        from prophet import Prophet
    except ImportError:
        return json.dumps({"error": "prophet not installed. pip install prophet"})

    try:
        df = load_dataframe(file_path, sheet_name=sheet_name if sheet_name != "" else 0)
        if date_col not in df.columns or target_col not in df.columns:
            return json.dumps({
                "error": f"Required columns missing. Need '{date_col}' and '{target_col}'. Got {list(df.columns)}"
            })

        df[date_col] = pd.to_datetime(df[date_col])
        pdf = df.rename(columns={date_col: "ds", target_col: "y"})[["ds", "y"]].copy()

        regressors = [c.strip() for c in regressor_cols.split(",") if c.strip()]
        for r in regressors:
            if r in df.columns:
                pdf[r] = df[r].astype(float).values

        pdf = pdf.dropna(subset=["y"]).sort_values("ds")

        m = Prophet(
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            daily_seasonality=daily,
        )
        for r in regressors:
            m.add_regressor(r)
        m.fit(pdf)

        future = m.make_future_dataframe(periods=horizon, freq=freq)
        # Fill regressors in future with last observed value (naive)
        for r in regressors:
            last_val = pdf[r].iloc[-1]
            future[r] = list(pdf[r].values) + [last_val] * horizon

        fc = m.predict(future)
        tail = fc.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

        return json.dumps({
            "model": "Prophet",
            "horizon": horizon,
            "freq": freq,
            "regressors_used": regressors,
            "dates": tail["ds"].dt.strftime("%Y-%m-%d").tolist(),
            "forecast": tail["yhat"].tolist(),
            "lower_95": tail["yhat_lower"].tolist(),
            "upper_95": tail["yhat_upper"].tolist(),
            "changepoints": [str(d) for d in m.changepoints.tolist()],
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def describe_prophet() -> str:
    """Return capabilities and when to use Prophet."""
    return json.dumps({
        "name": "Prophet (Meta)",
        "best_for": [
            "Strong seasonality (yearly, weekly, daily)",
            "Holiday effects, changepoints",
            "Missing data tolerance",
            "Business time series",
            "Extra regressors (multivariate)",
        ],
        "weak_at": ["Short series (<2 seasonal cycles)", "Sub-hourly high frequency", "Pure noise/random walks"],
        "needs": ["Datetime column ('ds')", "Numeric target ('y')"],
    })


if __name__ == "__main__":
    mcp.run(transport="stdio")
