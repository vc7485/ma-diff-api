import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
from fastapi import FastAPI
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI()

# Google Sheets setup
SHEET_NAME = "MA Diff Optimization"
SETTINGS_TAB = "Settings"
OUTPUT_TAB = "Output"
HEATMAP_TAB = "Heatmap"
EQUITY_CURVE_TAB = "Equity Curve"

SERVICE_ACCOUNT_FILE = "/etc/secrets/google_ma_diff_service_account"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
client = gspread.authorize(creds)

# Helper to read settings
def read_settings(sheet):
    rows = sheet.get_all_values()
    return {r[0].strip(): r[1].strip() for r in rows if len(r) >= 2}

# Download data
def fetch_yahoo_data(ticker, start_date, end_date, interval):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if data.empty:
        raise ValueError("No data found")
    data['Price'] = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    data.dropna(inplace=True)
    return data

# Generate trade signals
def generate_trade_signals(df, strategy_params, config):
    ma = int(strategy_params['ma_period'])
    threshold = strategy_params['diff_threshold']
    df['MovingAverage'] = df['Price'].rolling(window=ma, min_periods=1).mean()
    df['Diff'] = df['Price'] / df['MovingAverage'] - 1
    df['Signal'] = 0
    if config["execute_buy"]:
        df.loc[df['Diff'] > threshold, 'Signal'] = 1
    if config["execute_sell"]:
        df.loc[df['Diff'] < -threshold, 'Signal'] = -1
    df['Position'] = df['Signal'].shift(1, fill_value=0)
    df['Next_Open'] = df['Price'].shift(-1)
    df['Entry_Price'] = df['Next_Open'].shift(-1)
    df['Daily Return'] = df['Entry_Price'].pct_change().fillna(0)
    df['Strategy Return'] = df['Daily Return'] * df['Position']
    trade_executed = df['Position'].diff().fillna(0) != 0
    if config["use_percentage_cost"]:
        df.loc[trade_executed, 'Strategy Return'] -= config["percentage_cost"]
    else:
        df.loc[trade_executed, 'Strategy Return'] -= config["fixed_cost"] / config["initial_capital"]
    df['Cumulative Return'] = (1 + df['Strategy Return']).cumprod()
    df['Equity_Curve_Capital'] = df['Cumulative Return'] * config["initial_capital"]
    return df

# Optimization logic
def optimize_params(ma_period, diff_threshold, data, config):
    params = {"ma_period": ma_period, "diff_threshold": diff_threshold}
    df = generate_trade_signals(data.copy(), params, config)
    std = df['Strategy Return'].std()
    sharpe = (df['Strategy Return'].mean() / std) * np.sqrt(252) if std > 0 else np.nan
    annualized = df['Strategy Return'].mean() * 252
    max_dd = (df['Equity_Curve_Capital'].cummax() - df['Equity_Curve_Capital']).max()
    max_dd_pct = (max_dd / df['Equity_Curve_Capital'].cummax()).max() * 100
    calmar = annualized / abs(max_dd) if max_dd != 0 else 0
    final_capital = df['Equity_Curve_Capital'].iloc[-1]
    df['drawdown_end'] = df['Equity_Curve_Capital'].eq(df['Equity_Curve_Capital'].cummax())
    df['recovery_period'] = df.groupby(df['drawdown_end'].cumsum()).cumcount() + 1
    df['recovery_period'] = df['recovery_period'].where(~df['drawdown_end'], 0)
    MRP = df['recovery_period'].max()
    return {
        'MA Period': ma_period,
        'Diff Threshold': diff_threshold,
        'Sharpe Ratio': sharpe,
        'Annualized Return (%)': annualized * 100,
        'Max Drawdown ($)': max_dd,
        'Max Drawdown (%)': max_dd_pct,
        'Calmar Ratio': calmar,
        'Final Capital ($)': final_capital,
        'Maximum Recovery Period': MRP,
        'Equity Curve': df['Equity_Curve_Capital'].tolist(),
        'Dates': df.index.strftime('%Y-%m-%d').tolist()
    }

# API endpoint
@app.post("/run-backtest")
def run_backtest(sheet_id: dict):
    def background_job():
        sheet = client.open_by_key(sheet_id["sheet_id"])
        settings = read_settings(sheet.worksheet(SETTINGS_TAB))

        # Extract settings
        ticker = settings.get("Ticker")
        start_date = settings.get("Start Date")
        end_date = settings.get("End Date")
        timeframe = settings.get("Timeframe", "1d")
        capital = float(settings.get("Initial Capital", 1000))
        config = {
            "initial_capital": capital,
            "use_percentage_cost": settings.get("Use % Cost", "TRUE").upper() == "TRUE",
            "percentage_cost": float(settings.get("% Transaction Cost", 0.0006)),
            "fixed_cost": float(settings.get("Fixed Cost", 0)),
            "execute_buy": settings.get("Enable Buy", "TRUE").upper() == "TRUE",
            "execute_sell": settings.get("Enable Sell", "FALSE").upper() == "TRUE",
            "use_parallel": settings.get("Use Parallel", "TRUE").upper() == "TRUE"
        }

        use_opt = settings.get("Use Optimization", "TRUE").upper() == "TRUE"
        ma_min = int(settings.get("MA Min", 10))
        ma_max = int(settings.get("MA Max", 20))
        diff_min = float(settings.get("Diff Min", 0.01))
        diff_max = float(settings.get("Diff Max", 0.05))
        diff_step = float(settings.get("Diff Step", 0.002))

        data = fetch_yahoo_data(ticker, start_date, end_date, timeframe)
        param_grid = [(ma, round(diff, 3)) for ma in range(ma_min, ma_max+1)
                      for diff in np.arange(diff_min, diff_max+diff_step, diff_step)]

        results = []

        if config["use_parallel"]:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(optimize_params, ma, diff, data, config): (ma, diff)
                    for ma, diff in param_grid
                }
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
        else:
            for ma, diff in param_grid:
                result = optimize_params(ma, diff, data, config)
                results.append(result)

        df_results = pd.DataFrame(results)

        # Output top 10
        top10 = df_results.sort_values(by="Sharpe Ratio", ascending=False).head(10)
        sheet.worksheet(OUTPUT_TAB).clear()
        set_with_dataframe(sheet.worksheet(OUTPUT_TAB), top10)

        # Output Heatmap
        heatmap_data = df_results.pivot(index='MA Period', columns='Diff Threshold', values='Sharpe Ratio')
        sheet.worksheet(HEATMAP_TAB).clear()
        set_with_dataframe(sheet.worksheet(HEATMAP_TAB), heatmap_data)

        # Output Equity Curve
        best = df_results.sort_values(by="Sharpe Ratio", ascending=False).iloc[0]
        equity_curve = pd.DataFrame({
            "Date": best["Dates"],
            "Equity Curve Capital": best["Equity Curve"]
        })
        sheet.worksheet(EQUITY_CURVE_TAB).clear()
        set_with_dataframe(sheet.worksheet(EQUITY_CURVE_TAB), equity_curve)

    Thread(target=background_job).start()
    return {"message": "✅ Backtest started! Check Output, Heatmap & Equity Curve tabs shortly."}
