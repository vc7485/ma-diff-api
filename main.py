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

# === Google Sheets Setup ===
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
client = gspread.authorize(creds)
SHEET_ID = "YOUR_GOOGLE_SHEET_ID"
SHEET = client.open_by_key(SHEET_ID)

# === Settings Model from Google Sheets ===
class SettingsRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    use_optimization: bool
    use_precomputed_ma: bool
    ma_min: int
    ma_max: int
    diff_min: float
    diff_max: float
    diff_step: float

# === Strategy Logic ===
def ma_diff_strategy(df, ma_period, diff_threshold, trade_mode):
    df['MA'] = df['Close'].rolling(window=ma_period).mean()
    df['Diff'] = df['Close'] - df['MA']
    df['Signal'] = 0

    if trade_mode == "buy":
        df.loc[df['Diff'] > diff_threshold, 'Signal'] = 1
    elif trade_mode == "sell":
        df.loc[df['Diff'] < -diff_threshold, 'Signal'] = -1
    elif trade_mode == "both":
        df.loc[df['Diff'] > diff_threshold, 'Signal'] = 1
        df.loc[df['Diff'] < -diff_threshold, 'Signal'] = -1

    df['Position'] = df['Signal'].shift(1).fillna(0)
    df['Trade'] = df['Position'].diff().fillna(0)
    return df

def calculate_metrics(df, initial_capital=10000, transaction_cost=0.001):
    df['Returns'] = df['Position'].shift(1) * df['Close'].pct_change()
    df['Returns'] -= transaction_cost * df['Trade'].abs()
    df['Equity'] = (1 + df['Returns']).cumprod() * initial_capital

    roi = df['Equity'].iloc[-1] / initial_capital - 1
    annual_return = (df['Equity'].iloc[-1] / initial_capital) ** (252 / len(df)) - 1
    sharpe = df['Returns'].mean() / df['Returns'].std() * np.sqrt(252)
    mdd = ((df['Equity'] / df['Equity'].cummax()) - 1).min()
    calmar = annual_return / abs(mdd) if mdd != 0 else np.nan

    return {
        'Final Capital': df['Equity'].iloc[-1],
        'ROI': roi,
        'Annualized Return': annual_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': mdd,
        'Calmar Ratio': calmar
    }

def run_single_combo(df, ma, diff, trade_mode):
    df_copy = df.copy()
    df_copy = ma_diff_strategy(df_copy, ma, diff, trade_mode)
    metrics = calculate_metrics(df_copy)
    return {
        'MA Period': ma,
        'Diff Threshold': round(diff, 3),
        **{k: round(v, 4) for k, v in metrics.items()}
    }

# === Main Execution Endpoint ===
@app.post("/run-backtest")
def run_backtest(settings: SettingsRequest):
    print("📥 Downloading data...")
    df = yf.download(settings.ticker, start=settings.start_date, end=settings.end_date)
    df = df[['Close']].dropna()

    ma_list = range(settings.ma_min, settings.ma_max + 1)
    diff_list = np.arange(settings.diff_min, settings.diff_max + settings.diff_step, settings.diff_step)

    trade_modes = ["buy", "sell", "both"]
    all_results = {}

    for mode in trade_modes:
        print(f"🔁 Optimizing for {mode.upper()} mode...")

        combos = [(ma, diff) for ma in ma_list for diff in diff_list]

        def run_combo(i, combo):
            ma, diff = combo
            return run_single_combo(df, ma, diff, mode)

        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_combo, i, combo) for i, combo in enumerate(combos)]
            for f in futures:
                results.append(f.result())

        results_df = pd.DataFrame(results)
        top10_df = results_df.sort_values("Sharpe Ratio", ascending=False).head(10)
        sheet_name = f"Top 10 - {mode.capitalize()}"

        print(f"📤 Writing to sheet: {sheet_name}")
        try:
            SHEET.worksheet(sheet_name).clear()
        except:
            SHEET.add_worksheet(title=sheet_name, rows="100", cols="20")

        set_with_dataframe(SHEET.worksheet(sheet_name), top10_df)
        all_results[mode] = top10_df

    print("✅ All 3 Top 10 tabs updated.")
    return {"status": "success", "message": "Top 10 Buy, Sell, and Both results inserted into sheets."}
