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

# 🧾 Input model
class SettingsRequest(BaseModel):
    sheet_id: str
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
    else:  # both
        df['Signal'] = 0
        df.loc[df['Diff'] > diff_threshold, 'Signal'] = 1
        df.loc[df['Diff'] < -diff_threshold, 'Signal'] = -1

    df['Position'] = df['Signal'].shift(1).fillna(0)
    df['Trade'] = df['Position'].diff().fillna(0)
    return df

# 🔁 Run a single combo
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

# 🧪 Main Endpoint
@app.post("/run-backtest")
def run_backtest(settings: SettingsRequest):
    print("⚙️ Starting optimization...")

    df = yf.download(settings.ticker, start=settings.start_date, end=settings.end_date)
    df = df[['Close']].dropna()

    ma_list = range(settings.ma_min, settings.ma_max + 1)
    diff_list = np.arange(settings.diff_min, settings.diff_max + settings.diff_step, settings.diff_step)
    combos = [(ma, diff) for ma in ma_list for diff in diff_list]

    trade_modes = ["buy", "sell", "both"]
    sheet = client.open_by_key(settings.sheet_id)

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

    print("✅ Optimization complete!")
    return {"status": "success", "message": "Top 10 results for Buy/Sell/Both saved to Google Sheets."}
