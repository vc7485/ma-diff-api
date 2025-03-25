from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI()

# Google Sheets Setup
SETTINGS_TAB = "Settings"
TRADE_MODES = ["buy", "sell", "both"]
TAB_NAMES = {
    "buy": "Top 10 - Buy",
    "sell": "Top 10 - Sell",
    "both": "Top 10 - Both"
}
SERVICE_ACCOUNT_FILE = "/etc/secrets/google_ma_diff_service_account"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
client = gspread.authorize(creds)

# Read Settings
def read_settings(sheet):
    rows = sheet.get_all_values()
    return {r[0].strip(): r[1].strip() for r in rows if len(r) >= 2}

# Download Data
def fetch_yahoo_data(ticker, start_date, end_date, interval):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if df.empty:
        raise ValueError("No data found")
    df['Price'] = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
    df.dropna(inplace=True)
    return df

# Signal Generator
def generate_signals(df, ma_period, diff_threshold, config, trade_type):
    df['MA'] = df['Price'].rolling(window=ma_period).mean()
    df['Diff'] = df['Price'] / df['MA'] - 1
    df['Signal'] = 0

    if trade_type == "buy":
        df.loc[df['Diff'] > diff_threshold, 'Signal'] = 1
    elif trade_type == "sell":
        df.loc[df['Diff'] < -diff_threshold, 'Signal'] = -1
    else:
        df.loc[df['Diff'] > diff_threshold, 'Signal'] = 1
        df.loc[df['Diff'] < -diff_threshold, 'Signal'] = -1

    df['Position'] = df['Signal'].shift(1).fillna(0)
    df['Next_Open'] = df['Price'].shift(-1)
    df['Entry_Price'] = df['Next_Open'].shift(-1)
    df['Daily Return'] = df['Entry_Price'].pct_change(fill_method=None).fillna(0)
    df['Strategy Return'] = df['Daily Return'] * df['Position']

    trade_executed = df['Position'].diff().fillna(0) != 0
    if config["use_percentage_cost"]:
        df.loc[trade_executed, 'Strategy Return'] -= config["percentage_cost"]
    else:
        df.loc[trade_executed, 'Strategy Return'] -= config["fixed_cost"] / config["initial_capital"]

    df['Cumulative Return'] = (1 + df['Strategy Return']).cumprod()
    df['Equity'] = df['Cumulative Return'] * config["initial_capital"]
    return df

# Optimize One Combo
def optimize_params(ma, diff, df, config, trade_type):
    df = generate_signals(df.copy(), ma, diff, config, trade_type)
    std = df['Strategy Return'].std()
    sharpe = (df['Strategy Return'].mean() / std) * np.sqrt(252) if std > 0 else np.nan
    annualized = df['Strategy Return'].mean() * 252
    max_dd = (df['Equity'].cummax() - df['Equity']).max()
    max_dd_pct = (max_dd / df['Equity'].cummax()).max() * 100
    calmar = annualized / abs(max_dd) if max_dd != 0 else 0
    final_capital = df['Equity'].iloc[-1]

    df['drawdown_end'] = df['Equity'].eq(df['Equity'].cummax())
    df['recovery_period'] = df.groupby(df['drawdown_end'].cumsum()).cumcount() + 1
    df['recovery_period'] = df['recovery_period'].where(~df['drawdown_end'], 0)
    MRP = df['recovery_period'].max()

    return {
        "MA Period": ma,
        "Diff Threshold": round(diff, 3),
        "Sharpe Ratio": round(sharpe, 3),
        "Annualized Return (%)": round(annualized * 100, 3),
        "Max Drawdown ($)": round(max_dd, 3),
        "Max Drawdown (%)": round(max_dd_pct, 3),
        "Calmar Ratio": round(calmar, 3),
        "Final Capital ($)": round(final_capital, 3),
        "Maximum Recovery Period": int(MRP)
    }

# FastAPI Endpoint
@app.post("/run-backtest")
def run_backtest(sheet_id: dict):
    def background_job():
        sheet = client.open_by_key(sheet_id["sheet_id"])
        settings = read_settings(sheet.worksheet(SETTINGS_TAB))
        print("✅ Settings loaded.")

        config = {
            "initial_capital": float(settings.get("Initial Capital", 1000)),
            "use_percentage_cost": settings.get("Use % Cost", "TRUE").upper() == "TRUE",
            "percentage_cost": float(settings.get("% Transaction Cost", 0.0006)),
            "fixed_cost": float(settings.get("Fixed Cost", 0)),
            "use_parallel": settings.get("Use Parallel", "TRUE").upper() == "TRUE"
        }

        ticker = settings["Ticker"]
        start_date = settings["Start Date"]
        end_date = settings["End Date"]
        interval = settings.get("Timeframe", "1d")

        df = fetch_yahoo_data(ticker, start_date, end_date, interval)

        ma_range = np.arange(int(settings.get("MA Min", 10)), int(settings.get("MA Max", 20)) + 1, 1)
        diff_range = np.round(np.arange(float(settings.get("Diff Min", 0.01)), float(settings.get("Diff Max", 0.05)) + float(settings.get("Diff Step", 0.002)), float(settings.get("Diff Step", 0.002)), 3)
        param_grid = [(ma, diff) for ma in ma_range for diff in diff_range]

        for mode in TRADE_MODES:
            results = []
            if config["use_parallel"]:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {executor.submit(optimize_params, ma, diff, df, config, mode): (ma, diff) for ma, diff in param_grid}
                    for future in as_completed(futures):
                        results.append(future.result())
            else:
                for ma, diff in param_grid:
                    results.append(optimize_params(ma, diff, df, config, mode))

            result_df = pd.DataFrame(results).sort_values("Sharpe Ratio", ascending=False).head(10)

            tab_name = TAB_NAMES[mode]
            try:
                ws = sheet.worksheet(tab_name)
                ws.clear()
            except:
                ws = sheet.add_worksheet(title=tab_name, rows=100, cols=20)

            set_with_dataframe(ws, result_df)
            print(f"✅ {tab_name} written.")

    Thread(target=background_job).start()
    return {"message": "📊 Backtest started! Check it later."}
