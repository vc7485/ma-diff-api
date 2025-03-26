# Prepare Option A version of main.py as per user request (no Heatmap or Equity Curve yet)
# Includes only logic for Backtest Trade Type and writing to "Top 10" sheet

main_py_option_a = """
import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from typing import List

app = FastAPI()

class OptimizationRequest(BaseModel):
    sheet_id: str
    service_account_info: dict

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate / 252
    return_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
    return return_ratio * np.sqrt(252)

def get_sheet_config(sheet):
    config = {}
    data = sheet.get_all_values()
    for row in data:
        if len(row) >= 2:
            key = row[0].strip()
            value = row[1].strip()
            if key:
                config[key] = value
    return config

def run_backtest(ticker, start_date, end_date, ma_period, diff_threshold, trade_type, initial_capital, use_percentage_cost, percent_cost, fixed_cost):
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Open', 'Close']]
    df['MA'] = df['Close'].rolling(window=ma_period).mean()
    df['MA_Diff'] = df['Close'] - df['MA']

    if trade_type == "Buy":
        df['Signal'] = df['MA_Diff'] > diff_threshold
    elif trade_type == "Sell":
        df['Signal'] = df['MA_Diff'] < -diff_threshold
    elif trade_type == "Both":
        df['Signal'] = (df['MA_Diff'] > diff_threshold) | (df['MA_Diff'] < -diff_threshold)
    else:
        df['Signal'] = False

    df['Position'] = df['Signal'].shift(1).fillna(False)
    df['Trade'] = df['Position'] != df['Position'].shift(1)

    df['Returns'] = df['Close'].pct_change().fillna(0)
    df['Strategy_Returns'] = df['Returns'] * df['Position']

    if use_percentage_cost:
        df['Transaction_Cost'] = np.where(df['Trade'], percent_cost, 0)
    else:
        df['Transaction_Cost'] = np.where(df['Trade'], fixed_cost / df['Close'], 0)

    df['Net_Returns'] = df['Strategy_Returns'] - df['Transaction_Cost']
    df['Equity'] = (1 + df['Net_Returns']).cumprod() * initial_capital

    total_return = df['Equity'].iloc[-1] - initial_capital
    roi = (total_return / initial_capital) * 100
    sharpe_ratio = calculate_sharpe_ratio(df['Net_Returns'])
    max_drawdown = ((df['Equity'].cummax() - df['Equity']) / df['Equity'].cummax()).max() * 100

    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = ((df['Equity'].iloc[-1] / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    return {
        "MA Period": ma_period,
        "Diff Threshold": diff_threshold,
        "Sharpe Ratio": round(sharpe_ratio, 3),
        "Annualized Return": round(cagr, 3),
        "Max Drawdown": round(max_drawdown, 3),
        "Final Capital": round(df['Equity'].iloc[-1], 2)
    }

@app.post("/run-backtest")
def run_optimization(request: OptimizationRequest):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(request.service_account_info, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(request.sheet_id).worksheet("Settings")
    config = get_sheet_config(sheet)

    ticker = config.get("Ticker", "AAPL")
    start_date = config.get("Start Date", "2015-01-01")
    end_date = config.get("End Date", "2024-12-31")
    initial_capital = float(config.get("Initial Capital $", 1000))
    use_percentage_cost = config.get("Use % Cost", "TRUE").upper() == "TRUE"
    percent_cost = float(config.get("% Transaction Cost", 0.0))
    fixed_cost = float(config.get("Fixed Cost $", 0.0))
    trade_type = config.get("Backtest Trade Type", "Buy")

    ma_min = int(config.get("MA Min", 5))
    ma_max = int(config.get("MA Max", 50))
    diff_min = float(config.get("Diff Min", 0.0))
    diff_max = float(config.get("Diff Max", 10.0))
    diff_step = float(config.get("Diff Step", 0.5))

    results = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for ma in range(ma_min, ma_max + 1):
            diff = diff_min
            while diff <= diff_max:
                futures.append(executor.submit(run_backtest, ticker, start_date, end_date, ma, round(diff, 4),
                                               trade_type, initial_capital, use_percentage_cost, percent_cost, fixed_cost))
                diff += diff_step

        for future in futures:
            results.append(future.result())

    top_10_results = sorted(results, key=lambda x: x["Sharpe Ratio"], reverse=True)[:10]

    output_sheet_name = "Top 10"
    try:
        output_sheet = client.open_by_key(request.sheet_id).worksheet(output_sheet_name)
        output_sheet.clear()
    except:
        output_sheet = client.open_by_key(request.sheet_id).add_worksheet(title=output_sheet_name, rows="100", cols="20")

    headers = list(top_10_results[0].keys())
    values = [headers] + [[res[h] for h in headers] for res in top_10_results]
    output_sheet.update("A1", values)

    return {"status": "✅ Top 10 result updated based on trade type: " + trade_type}
"""

with open("/mnt/data/main_option_a.py", "w") as f:
    f.write(main_py_option_a)

"/mnt/data/main_option_a.py"

