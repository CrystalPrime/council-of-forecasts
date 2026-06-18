![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-blueviolet?logo=langchain)
![LangChain](https://img.shields.io/badge/LangChain-Agents-green?logo=langchain)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-orange?logo=groq)
![MCP](https://img.shields.io/badge/MCP-3_Servers-lightblue)
![statsmodels](https://img.shields.io/badge/statsmodels-ARIMA%2FSARIMA-blue)
![Prophet](https://img.shields.io/badge/Prophet-Meta-red)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

# Council of Forecasts — Multi-Agent Time Series Forecasting

A LangGraph orchestration layer that routes time series forecasting jobs to three specialist **MCP servers** — ARIMA/SARIMA, Prophet, and LSTM — each running as an independent stdio process. Before any forecast is computed, a built-in statistical Tester analyzes the dataset (stationarity, seasonality, trend, multivariate correlation) and recommends the best model. The user approves, then the forecast runs.

## Architecture

```
manager/agent.py  (LangGraph ReAct Agent — Groq llama-3.3-70b)
    │
    ├── Tester Tools (built-in, no MCP)
    │       ├── inspect_dataset          — shape, dtypes, missing values
    │       ├── test_stationarity        — ADF + KPSS tests
    │       ├── detect_seasonality       — ACF on detrended series, additive/multiplicative
    │       ├── detect_trend             — Mann-Kendall + linear regression
    │       └── check_multivariate_correlation — Pearson, strong/weak predictors
    │
    └── MCP Servers (stdio subprocesses)
            ├── arima_server.py  →  forecast_arima   (ARIMA / SARIMA / SARIMAX)
            ├── prophet_server.py → forecast_prophet  (Meta Prophet + regressors)
            └── lstm_server.py   →  forecast_lstm    (PyTorch LSTM, multivariate)
```

## Workflow

```
User: "data/sales.xlsx dosyasını analiz et ve forecast yap"

1. Agent calls inspect_dataset        → learns shape, columns, file format
2. Agent calls test_stationarity      → ADF p=0.72 → non-stationary
3. Agent calls detect_seasonality     → period=12, strength=0.89, multiplicative
4. Agent calls detect_trend           → significant upward trend (τ=0.61)
5. Agent calls check_multivariate_correlation → marketing_spend r=0.54

Agent recommends: "Prophet (multiplicative seasonality + strong trend).
Horizon kaç adım olsun?"

User: "12 ay"

6. Agent calls forecast_prophet(horizon=12, regressor_cols="marketing_spend")
7. Agent summarizes: trend direction, first 10 forecast values, confidence bounds
```

## Features

- **Semi-autonomous** — agent inspects, tests, and recommends without prompting; user only approves the model and horizon
- **Statistical rigor** — model selection is grounded in ADF/KPSS, ACF peaks, Mann-Kendall, and correlation analysis, not LLM intuition alone
- **Multivariate support** — SARIMAX with exogenous regressors, Prophet `add_regressor`, multi-feature LSTM input
- **CSV + Excel** — all tools accept `.csv`, `.tsv`, `.xlsx`, `.xls` via shared `common/data_loader.py`
- **Modular MCP servers** — each forecaster is an independent process; plug into any MCP-compatible client

## Model selection heuristics

| Test result | Recommendation |
|---|---|
| Stationary + no seasonality + short series | ARIMA |
| Non-stationary + monthly seasonality | SARIMA |
| Strong seasonality + datetime + business data | Prophet |
| Multiplicative seasonality (growing amplitude) | Prophet (log mode) |
| Long series (500+ rows) + multivariate | LSTM |
| Strong exogenous predictors | SARIMAX or Prophet + regressors |

## Setup

**1. Clone and install**

```bash
git clone https://github.com/CrystalPrime/council-of-forecasts
cd council-of-forecasts
pip install -r requirements.txt
```

**2. Set environment variables**

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile   # optional
```

Get a free Groq key at [console.groq.com](https://console.groq.com)

**3. Run**

```bash
python manager/agent.py
```

Then drop in a file path:

```
You: data/sample_sales.csv dosyasını analiz et
```

The agent will inspect → test → recommend → wait for your approval → forecast → summarize.

## Project structure

```
council-of-forecasts/
├── manager/
│   ├── agent.py          # LangGraph ReAct agent + MCP client + chat loop
│   └── tester_tools.py   # Statistical analysis tools (Tester role)
├── servers/
│   ├── arima_server.py   # ARIMA/SARIMA/SARIMAX MCP server
│   ├── prophet_server.py # Meta Prophet MCP server
│   └── lstm_server.py    # PyTorch LSTM MCP server
├── common/
│   ├── __init__.py       # Package exports
│   └── data_loader.py    # Shared CSV/TSV/XLSX/XLS loader with path normalization
├── data/
│   └── sample_sales.csv  # Synthetic seasonal + multivariate test data (48 months)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Stack

- **LangGraph** — ReAct agent orchestration with MemorySaver
- **LangChain** — Tool definitions, Groq integration
- **Groq** — LLM inference (llama-3.3-70b-versatile)
- **MCP (stdio)** — Three independent forecast servers
- **statsmodels** — ARIMA, SARIMA, SARIMAX, ADF, KPSS, ACF
- **Prophet** — Meta's time series forecasting library
- **PyTorch** — LSTM implementation
- **pandas / scipy** — Data loading, Mann-Kendall trend test

---

*Part of an ongoing series of LangChain and LangGraph projects — see also [Daily Briefing](https://github.com/CrystalPrime/daily-briefing) and [FinSight](https://github.com/CrystalPrime/FinSight).*
