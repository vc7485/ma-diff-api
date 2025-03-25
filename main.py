from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# 🔐 Google Sheets Credentials
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
client = gspread.authorize(creds)

# 🔧 Sheet Request (only need sheet_id from Google Apps Script)
class SettingsRequest(BaseModel):
    sheet_id: str

# 📈 Metric calculations
def calculate_metrics(df, initial_capital=10000, transaction_cost=0.001):
    df['Returns'] = df['Position'].shift(1) * df['Close'].pct_change()
    df['Returns'] -= transaction_cost * df['Trade'].fillna(0).abs()
    df['Equity'] = (1 + df['Returns']).cumprod() * initial_capital

    roi = df['Equity'].iloc[-1] / initial_capital - 1
    annualized_return = (df['Equity'].iloc[-1] / initial_capital) ** (252 / len(df)) - 1
    sharpe_ratio = df['Returns'].mean() / df['Returns'].std() * np.sqrt(252)
    max_drawdown = ((df['Equity'] / df['Equity'].cummax()) - 1).min()
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        'Final Capital': df['Equity'].iloc[-1],
        'ROI': roi,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio
    }

# 🧠 Strategy logic
def ma_diff_strategy(df, ma_period, diff_threshold, trade_type="both"):
    df['MA'] = df['Close'].rolling(window=ma_period).mean()
    df['Diff'] = df['Close'] - df['MA']

    if trade_type == "buy":
        df['Signal'] = (df['Diff'] > diff_threshold).astype(int)
    elif trade_type == "sell":
        df['Signal'] = (df['Diff'] < -diff_threshold).astype(int) * -1
    else:
        df['Signal'] = 0
        df.loc[df['Diff'] > diff_threshold, 'Signal'] = 1
        df.loc[df['Diff'] < -diff_threshold, 'Signal'] = -1

    df['Position'] = df['Signal'].shift(1).fillna(0)
    df['Trade'] = df['Position'].diff().fillna(0)
    return df

# 🔁 Single combo run
def run_combination(df, ma, diff, trade_type):
    df_copy = df.copy()
    df_copy = ma_diff_strategy(df_copy, ma, diff, trade_type)
    metrics = calculate_metrics(df_copy)
    return {
        'MA Period': ma,
        'Diff Threshold': round(diff, 3),
        **{k: round(v, 3) for k, v in metrics.items()},
        'Equity Curve': df_copy['Equity'].values,
        'Date': df_copy.index
    }

# 📥 Pull settings from Google Sheet tab
def read_settings_from_sheet(sheet):
    ws = sheet.worksheet("Settings")
    df = pd.DataFrame(ws.get_all_records())
    row = df.iloc[0]
    return {
        'ticker': row['ticker'],
        'start_date': row['start_date'],
        'end_date': row['end_date'],
        'use_optimization': row['use_optimization'],
        'use_precomputed_ma': row['use_precomputed_ma'],
        'ma_min': int(row['ma_min']),
        'ma_max': int(row['ma_max']),
        'diff_min': float(row['diff_min']),
        'diff_max': float(row['diff_max']),
        'diff_step': float(row['diff_step'])
    }

# 🚀 Main API Endpoint
@app.post("/run-backtest")
def run_backtest(request: SettingsRequest):
    print("📥 Reading sheet settings...")
    sheet = client.open_by_key(request.sheet_id)
    config = read_settings_from_sheet(sheet)

    df = yf.download(config['ticker'], start=config['start_date'], end=config['end_date'])
    df = df[['Close']].dropna()

    ma_list = range(config['ma_min'], config['ma_max'] + 1)
    diff_list = np.arange(config['diff_min'], config['diff_max'] + config['diff_step'], config['diff_step'])
    combos = [(ma, diff) for ma in ma_list for diff in diff_list]

    trade_modes = ["buy", "sell", "both"]

    for mode in trade_modes:
        print(f"🔄 Optimizing {mode}...")

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda args: run_combination(df, *args, mode), combos))

        result_df = pd.DataFrame(results).drop(columns=["Equity Curve", "Date"])
        top10 = result_df.sort_values("Sharpe Ratio", ascending=False).head(10)

        tab_name = f"Top 10 - {mode.capitalize()}"
        try:
            ws = sheet.worksheet(tab_name)
            ws.clear()
        except:
            ws = sheet.add_worksheet(title=tab_name, rows=100, cols=20)

        set_with_dataframe(ws, top10)

    print("✅ Done!")
    return {"status": "success", "message": "Top 10 Buy/Sell/Both strategies saved to sheet."}
