from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import io
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
from concurrent.futures import ThreadPoolExecutor
import datetime
import time

app = FastAPI()

# Google Sheets Setup
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
client = gspread.authorize(creds)

# Google Sheet ID
SHEET_ID = "YOUR_GOOGLE_SHEET_ID"
SHEET = client.open_by_key(SHEET_ID)

# Request Model
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

def calculate_metrics(df, initial_capital=10000, transaction_cost=0.001):
    df['Returns'] = df['Position'].shift(1) * df['Close'].pct_change()
    df['Returns'] -= transaction_cost * df['Trade'].fillna(0).abs()
    df['Equity'] = (1 + df['Returns']).cumprod() * initial_capital

    roi = df['Equity'].iloc[-1] / initial_capital - 1
    annualized_return = (df['Equity'].iloc[-1] / initial_capital) ** (252 / len(df)) - 1
    sharpe_ratio = df['Returns'].mean() / df['Returns'].std() * np.sqrt(252)
    max_drawdown = ((df['Equity'] / df['Equity'].cummax()) - 1).min()
    calmar_ratio = annualized_return / abs(max_drawdown)

    return {
        'Final Capital': df['Equity'].iloc[-1],
        'ROI': roi,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio
    }

def ma_diff_strategy(df, ma_period, diff_threshold):
    df['MA'] = df['Close'].rolling(window=ma_period).mean()
    df['Diff'] = df['Close'] - df['MA']
    df['Signal'] = (df['Diff'] > diff_threshold).astype(int)
    df['Position'] = df['Signal'].shift(1).fillna(0)
    df['Trade'] = df['Position'].diff().fillna(0)
    return df

def run_single_combination(df, ma_period, diff_threshold):
    df_copy = df.copy()
    df_copy = ma_diff_strategy(df_copy, ma_period, diff_threshold)
    metrics = calculate_metrics(df_copy)
    return {
        'MA Period': ma_period,
        'Diff Threshold': round(diff_threshold, 3),
        **{k: round(v, 3) for k, v in metrics.items()},
        'Equity Curve': df_copy['Equity'].values,
        'Date': df_copy.index
    }

@app.post("/run-backtest")
def run_backtest(settings: SettingsRequest):
    print("⚙️ Starting optimization...")

    df = yf.download(settings.ticker, start=settings.start_date, end=settings.end_date)
    df = df[['Close']].dropna()

    ma_list = range(settings.ma_min, settings.ma_max + 1)
    diff_list = np.arange(settings.diff_min, settings.diff_max + settings.diff_step, settings.diff_step)

    combinations = [(ma, diff) for ma in ma_list for diff in diff_list]
    results = []
    progress_intervals = max(1, len(combinations) // 10)

    def run_and_log(i, combo):
        ma, diff = combo
        result = run_single_combination(df, ma, diff)
        if i % progress_intervals == 0:
            print(f"🔄 Progress: {round(i/len(combinations)*100)}%")
        return result

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_and_log, i, combo) for i, combo in enumerate(combinations)]
        for f in futures:
            results.append(f.result())

    results_df = pd.DataFrame(results)

    best_row = results_df.sort_values("Sharpe Ratio", ascending=False).iloc[0]
    best_curve = best_row['Equity Curve']
    best_dates = best_row['Date']

    # Write to Google Sheets
    print("📤 Writing results to Google Sheets...")
    SHEET.worksheet("Settings").clear()
    set_with_dataframe(SHEET.worksheet("Settings"), pd.DataFrame([settings.dict()]))

    output_df = results_df.drop(columns=["Equity Curve", "Date"])
    SHEET.worksheet("Output").clear()
    set_with_dataframe(SHEET.worksheet("Output"), output_df)

    # Create Heatmap
    heatmap = results_df.pivot("MA Period", "Diff Threshold", "Sharpe Ratio")
    SHEET.worksheet("Heatmap").clear()
    set_with_dataframe(SHEET.worksheet("Heatmap"), heatmap)

    # Create Equity Curve
    equity_df = pd.DataFrame({"Date": best_dates, "Equity": best_curve})
    SHEET.worksheet("Equity Curve").clear()
    set_with_dataframe(SHEET.worksheet("Equity Curve"), equity_df)

    print("✅ Optimization complete!")
    return {"status": "success", "top_result": best_row.to_dict()}
