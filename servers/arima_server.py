"""
ARIMA/SARIMA MCP Server
Univariate & multivariate (SARIMAX with exogenous regressors) forecasting.
Supports CSV, TSV, XLSX, XLS input.
"""
import sys
from pathlib import Path

# Make sibling `common/` package importable when running as a stdio subprocess
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
import json

from common import load_dataframe

warnings.filterwarnings("ignore")

mcp = FastMCP("arima-forecast")


def _load_series(file_path: str, target_col: str, date_col: str | None, sheet_name):
    df = load_dataframe(file_path, sheet_name=sheet_name if sheet_name != "" else 0)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in file. Columns: {list(df.columns)}")
    return df


def _auto_order(series: pd.Series, seasonal: bool, m: int):
    """Simple grid search for (p,d,q) and optionally (P,D,Q,m)."""
    # Differencing degree from ADF
    d = 0
    s = series.dropna().copy()
    for _ in range(2):
        if adfuller(s)[1] < 0.05:
            break
        s = s.diff().dropna()
        d += 1

    best_aic = np.inf
    best_order = (1, d, 1)
    best_seasonal = (0, 0, 0, 0)

    p_range = range(0, 3)
    q_range = range(0, 3)
    for p in p_range:
        for q in q_range:
            try:
                if seasonal:
                    for P in range(0, 2):
                        for Q in range(0, 2):
                            try:
                                model = SARIMAX(
                                    series,
                                    order=(p, d, q),
                                    seasonal_order=(P, 1, Q, m),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                ).fit(disp=False)
                                if model.aic < best_aic:
                                    best_aic = model.aic
                                    best_order = (p, d, q)
                                    best_seasonal = (P, 1, Q, m)
                            except Exception:
                                continue
                else:
                    model = SARIMAX(
                        series,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_order = (p, d, q)
            except Exception:
                continue
    return best_order, best_seasonal, best_aic


@mcp.tool()
def forecast_arima(
    file_path: str,
    target_col: str,
    date_col: str = "",
    horizon: int = 12,
    seasonal: bool = False,
    seasonal_period: int = 12,
    exog_cols: str = "",
    sheet_name: str = "",
) -> str:
    """
    Fit ARIMA/SARIMA/SARIMAX and forecast `horizon` steps ahead.

    Args:
        file_path: Path to CSV, TSV, XLSX, or XLS file.
        target_col: Column to forecast.
        date_col: Datetime column name (optional, sets index).
        horizon: Number of future steps to predict.
        seasonal: If True, fit SARIMA.
        seasonal_period: Seasonal period m (e.g. 12 for monthly).
        exog_cols: Comma-separated exogenous regressor columns (SARIMAX). Empty if none.
        sheet_name: For Excel files. Empty = first sheet.

    Returns:
        JSON string with: order, seasonal_order, aic, forecast (list), conf_int.
    """
    try:
        df = _load_series(file_path, target_col, date_col or None, sheet_name)
        y = df[target_col].astype(float).dropna()

        exog = None
        exog_future = None
        exog_list = [c.strip() for c in exog_cols.split(",") if c.strip()]
        if exog_list:
            exog = df[exog_list].astype(float).loc[y.index]
            # Use last observed exog values, repeated, as naive future exog
            exog_future = pd.DataFrame(
                np.tile(exog.iloc[-1].values, (horizon, 1)),
                columns=exog_list,
            )

        order, seasonal_order, aic = _auto_order(y, seasonal, seasonal_period)

        model = SARIMAX(
            y,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order if seasonal else (0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        fc = model.get_forecast(steps=horizon, exog=exog_future)
        mean = fc.predicted_mean.tolist()
        ci = fc.conf_int(alpha=0.05).values.tolist()

        return json.dumps({
            "model": "SARIMAX" if seasonal or exog_list else "ARIMA",
            "order": list(order),
            "seasonal_order": list(seasonal_order) if seasonal else None,
            "aic": float(aic),
            "horizon": horizon,
            "forecast": mean,
            "conf_int_95": ci,
            "exog_used": exog_list,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def describe_arima() -> str:
    """Return capabilities and when to use ARIMA/SARIMA/SARIMAX."""
    return json.dumps({
        "name": "ARIMA/SARIMA/SARIMAX",
        "best_for": [
            "Stationary or differenceable series",
            "Short-to-medium horizons",
            "Clear linear autocorrelation",
            "Optional seasonality (SARIMA)",
            "Optional exogenous regressors (SARIMAX, multivariate)",
        ],
        "weak_at": ["Long horizons", "Highly nonlinear dynamics", "Multiple regime shifts"],
        "needs": ["1D target time series", "Optional datetime index", "Optional exog columns"],
    })


if __name__ == "__main__":
    mcp.run(transport="stdio")
