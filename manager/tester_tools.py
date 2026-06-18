"""
Tester tools — data analysis functions exposed as LangChain tools.
The Manager agent calls these to understand the dataset before recommending a model.
Supports CSV, TSV, XLSX, XLS via the shared loader.
"""
import sys
from pathlib import Path

# Make sibling `common/` package importable when running from manager/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import tool
import pandas as pd
import numpy as np
import json

from common import load_dataframe, list_sheets, SUPPORTED_EXTS


def _load(file_path: str, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Internal wrapper so each tool stays compact. Path normalization & existence
    checks are handled inside load_dataframe (strips quotes/whitespace)."""
    return load_dataframe(file_path, sheet_name=sheet_name)


@tool
def inspect_dataset(file_path: str, sheet_name: str = "") -> str:
    """
    Inspect a dataset file: shape, columns, dtypes, missing values, head samples.
    Supports CSV, TSV, XLSX, XLS. Call this FIRST when the user provides a new dataset.

    Args:
        file_path: Path to .csv, .tsv, .xlsx, or .xls file.
        sheet_name: For Excel only — sheet name. Empty = first sheet (default).
                    If multiple sheets exist, this tool also reports all sheet names.
    """
    try:
        sheets = list_sheets(file_path)  # empty list for CSV
        sheet = sheet_name if sheet_name else 0
        df = _load(file_path, sheet_name=sheet)

        result = {
            "file_format": Path(file_path).suffix.lower(),
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "missing": {c: int(df[c].isna().sum()) for c in df.columns},
            "head": df.head(3).to_dict(orient="records"),
            "numeric_summary": df.describe().to_dict() if len(df.select_dtypes('number').columns) else {},
        }
        if sheets:
            result["available_sheets"] = sheets
            result["active_sheet"] = sheet_name or sheets[0]
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def test_stationarity(file_path: str, target_col: str, sheet_name: str = "") -> str:
    """
    Run ADF and KPSS tests on the target column. Supports CSV/TSV/XLSX/XLS.
    Returns p-values and stationarity verdict.
    """
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        df = _load(file_path, sheet_name=sheet_name or 0)
        s = df[target_col].astype(float).dropna()

        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")

        adf_stationary = adf_p < 0.05
        kpss_stationary = kpss_p > 0.05

        if adf_stationary and kpss_stationary:
            verdict = "stationary"
        elif not adf_stationary and not kpss_stationary:
            verdict = "non-stationary (needs differencing)"
        else:
            verdict = "borderline / trend-stationary"

        return json.dumps({
            "adf_pvalue": float(adf_p),
            "kpss_pvalue": float(kpss_p),
            "verdict": verdict,
            "recommended_d": 0 if adf_stationary else 1,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def detect_seasonality(
    file_path: str, target_col: str, date_col: str = "", sheet_name: str = ""
) -> str:
    """
    Detect seasonality via ACF peaks on detrended series and decomposition strength.
    Tries both additive and multiplicative decomposition. Supports CSV/TSV/XLSX/XLS.
    """
    try:
        from statsmodels.tsa.stattools import acf
        from statsmodels.tsa.seasonal import seasonal_decompose

        df = _load(file_path, sheet_name=sheet_name or 0)
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).set_index(date_col)
        s = df[target_col].astype(float).dropna()

        n = len(s)
        max_lag = min(n // 2 - 1, 60)
        if max_lag < 4:
            return json.dumps({"error": "Series too short for seasonality detection"})

        # Detrend before ACF so trend doesn't mask seasonal peaks.
        s_detrended = s.diff().dropna()
        acf_vals = acf(s_detrended, nlags=min(max_lag, len(s_detrended) - 2), fft=True)
        acf_raw = acf(s, nlags=max_lag, fft=True)

        candidates = []
        for lag in range(2, len(acf_vals) - 1):
            if (
                acf_vals[lag] > acf_vals[lag - 1]
                and acf_vals[lag] > acf_vals[lag + 1]
                and acf_vals[lag] > 0.2
            ):
                candidates.append((lag, float(acf_vals[lag])))
        candidates.sort(key=lambda x: -x[1])

        # Snap to common business periods if close
        common_periods = [7, 12, 24, 52, 4]
        preferred = None
        for cp in common_periods:
            for lag, val in candidates:
                if abs(lag - cp) <= 1 and val > 0.15:
                    preferred = cp
                    break
            if preferred:
                break

        dominant = preferred if preferred else (candidates[0][0] if candidates else None)

        strength = None
        best_mode = None
        if dominant and n >= 2 * dominant:
            for mode in ("additive", "multiplicative"):
                try:
                    if mode == "multiplicative" and (s <= 0).any():
                        continue
                    dec = seasonal_decompose(s, period=dominant, model=mode, extrapolate_trend="freq")
                    resid = dec.resid.dropna()
                    seasonal_plus_resid = (dec.seasonal + dec.resid).dropna()
                    var_resid = float(np.var(resid))
                    var_sr = float(np.var(seasonal_plus_resid))
                    if var_sr > 0:
                        sc = max(0.0, 1.0 - var_resid / var_sr)
                        if strength is None or sc > strength:
                            strength = sc
                            best_mode = mode
                except Exception:
                    continue

        return json.dumps({
            "dominant_period": dominant,
            "seasonality_strength": strength,
            "decomposition_mode": best_mode,
            "top_acf_lags_detrended": candidates[:5],
            "raw_acf_at_period": float(acf_raw[dominant]) if dominant and dominant < len(acf_raw) else None,
            "has_seasonality": bool(dominant and (strength or 0) > 0.3),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def detect_trend(file_path: str, target_col: str, sheet_name: str = "") -> str:
    """
    Detect monotonic trend via Mann-Kendall and linear regression slope.
    Supports CSV/TSV/XLSX/XLS.
    """
    try:
        from scipy.stats import kendalltau, linregress
        df = _load(file_path, sheet_name=sheet_name or 0)
        s = df[target_col].astype(float).dropna().values
        t = np.arange(len(s))

        tau, p_kendall = kendalltau(t, s)
        slope, intercept, r, p_lin, _ = linregress(t, s)

        if p_kendall < 0.05 and abs(tau) > 0.1:
            direction = "upward" if tau > 0 else "downward"
            verdict = f"significant {direction} trend"
        else:
            verdict = "no significant trend"

        return json.dumps({
            "kendall_tau": float(tau),
            "kendall_pvalue": float(p_kendall),
            "linear_slope": float(slope),
            "linear_r2": float(r ** 2),
            "verdict": verdict,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def check_multivariate_correlation(
    file_path: str, target_col: str, sheet_name: str = ""
) -> str:
    """
    Pearson correlation between target and other numeric columns.
    Helps choose exogenous regressors for multivariate forecasting.
    Supports CSV/TSV/XLSX/XLS.
    """
    try:
        df = _load(file_path, sheet_name=sheet_name or 0)
        num = df.select_dtypes("number")
        if target_col not in num.columns:
            return json.dumps({"error": f"{target_col} is not numeric"})
        corr = num.corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        return json.dumps({
            "correlations_with_target": corr.to_dict(),
            "strong_predictors": corr[abs(corr) > 0.5].index.tolist(),
            "weak_predictors": corr[abs(corr) < 0.2].index.tolist(),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


TESTER_TOOLS = [
    inspect_dataset,
    test_stationarity,
    detect_seasonality,
    detect_trend,
    check_multivariate_correlation,
]
