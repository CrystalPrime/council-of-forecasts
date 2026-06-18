"""
Council of Forecasts — Manager Agent
- Groq-powered LangGraph agent
- Built-in Tester tools (statistical data analysis)
- Connects to 3 MCP servers: ARIMA, Prophet, LSTM
- Semi-autonomous: analyzes data, recommends a model, waits for user approval, runs forecast.
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Windows fix: MCP stdio servers spawn subprocesses; ProactorEventLoop (default on Win + Py3.8+)
# leaks subprocess pipe transports on shutdown and prints "Event loop is closed" tracebacks.
# SelectorEventLoop handles cleanup cleanly. Must be set BEFORE any asyncio code runs.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from tester_tools import TESTER_TOOLS

load_dotenv()

SERVERS_DIR = Path(__file__).resolve().parent.parent / "servers"

SYSTEM_PROMPT = """You are the Council of Forecasts — the chief forecasting strategist.

You orchestrate three specialist forecasters (ARIMA, Prophet, LSTM) and have a built-in
statistical Tester. Your workflow when a user provides a dataset (CSV path):

1. **Inspect**: call `inspect_dataset` to learn shape, columns, missing data.
2. **Identify**: ask the user which column is the target and which is the date column
   IF NOT CLEAR. If obvious from column names (e.g. "value" + "date"), proceed.
3. **Analyze (Tester role)**: run the relevant tests:
   - `test_stationarity` — for ARIMA decisions
   - `detect_seasonality` — for SARIMA / Prophet decisions
   - `detect_trend` — overall trend
   - `check_multivariate_correlation` — if multiple numeric columns exist
4. **Recommend**: based on test results, recommend ONE model with reasoning.
   Decision heuristics:
   - Stationary + no seasonality + short series → ARIMA
   - Seasonality + datetime + business series → Prophet
   - Long series (500+) + multivariate + nonlinear → LSTM
   - Borderline cases: explain trade-offs, suggest top 2
5. **Wait for approval**: present the recommendation and ask the user to confirm
   OR choose a different model. Do NOT call a forecast tool before confirmation.
6. **Forecast**: once user confirms, call the corresponding tool:
   - `forecast_arima` (set seasonal=True if seasonality detected; pass exog_cols for multivariate)
   - `forecast_prophet` (pass regressor_cols for multivariate)
   - `forecast_lstm` (pass feature_cols for multivariate)
7. **Report**: summarize the forecast in plain language: trend direction, confidence,
   any caveats. Show the first 5–10 forecast values.

Rules:
- Always answer in the user's language (Turkish if they write in Turkish).
- Be concise. No filler.
- If a tool returns an error, explain it and suggest a fix.
- If the user just says "forecast this CSV", proceed autonomously through steps 1–5,
  then stop and ask for approval before step 6.
"""


async def build_agent():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing in .env")

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.2,
        api_key=api_key,
    )

    # MCP servers
    mcp_client = MultiServerMCPClient({
        "arima": {
            "command": "python",
            "args": [str(SERVERS_DIR / "arima_server.py")],
            "transport": "stdio",
        },
        "prophet": {
            "command": "python",
            "args": [str(SERVERS_DIR / "prophet_server.py")],
            "transport": "stdio",
        },
        "lstm": {
            "command": "python",
            "args": [str(SERVERS_DIR / "lstm_server.py")],
            "transport": "stdio",
        },
    })

    mcp_tools = await mcp_client.get_tools()
    all_tools = TESTER_TOOLS + mcp_tools

    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )
    return agent


async def chat_loop():
    agent = await build_agent()
    config = {"configurable": {"thread_id": "council-session"}}

    print("=" * 60)
    print("  Council of Forecasts — Manager Agent")
    print("  Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() in {"quit", "exit"}:
            break

        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        messages = result["messages"]
        last_human_idx = max(
            (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
            default=-1,
        )
        answer = "No reply."
        for m in messages[last_human_idx + 1:]:
            if isinstance(m, AIMessage) and m.content and m.content.strip():
                answer = m.content
        print(f"\nCouncil: {answer}\n")


if __name__ == "__main__":
    asyncio.run(chat_loop())
