import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import glob

START_DATE = '2016-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
FUTURE_YEARS = 5
FUTURE_DAYS = FUTURE_YEARS * 365

TICKERS = {
    'Gold': 'GC=F',
    'Bitcoin': 'BTC-USD'
}

def download_data(ticker, label):
    data = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"Failed to download data for {label} ({ticker}). Try again later.")
    series = data['Adj Close']
    series.name = label
    return series

def date_to_num(dates):
    return np.array([date.toordinal() for date in dates])

def forecast_polyfit(dates, prices, future_days, degree=2):
    x = date_to_num(dates)
    y = prices.values
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)

    last_day = dates[-1]
    future_dates = [last_day + timedelta(days=i) for i in range(1, future_days + 1)]
    x_future = date_to_num(future_dates)
    y_future = poly(x_future)

    return future_dates, y_future

def main():
    gold_series = download_data(TICKERS['Gold'], 'Gold')
    btc_series = download_data(TICKERS['Bitcoin'], 'Bitcoin')

    df = pd.concat([gold_series, btc_series], axis=1).dropna()
    df.columns = ['Gold', 'Bitcoin']

    gold_future_dates, gold_forecast = forecast_polyfit(df.index, df['Gold'], FUTURE_DAYS)
    btc_future_dates, btc_forecast = forecast_polyfit(df.index, df['Bitcoin'], FUTURE_DAYS)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#2c2c2c')
    ax.set_facecolor('#2c2c2c')

    ax.plot(df.index, df['Gold'], label='Gold (actual)', color='gold')
    ax.plot(df.index, df['Bitcoin'], label='Bitcoin (actual)', color='orange')
    ax.plot(gold_future_dates, gold_forecast, '--', label='Gold (forecast)', color='darkgoldenrod')
    ax.plot(btc_future_dates, btc_forecast, '--', label='Bitcoin (forecast)', color='red')

    today_str = datetime.today().strftime('%Y-%m-%d')
    ax.set_title(f"Bitcoin vs Gold Forecast\nFrom 2016 to {today_str}, with projection to 2030",
                 fontsize=14, color='white')
    ax.set_xlabel('Date', color='white')
    ax.set_ylabel('Price (USD)', color='white')
    ax.tick_params(colors='white')
    ax.set_ylim(0, 220000)

    # Добавляем подпись (водяной знак)
    ax.text(df.index[-1], 2000, 'Forecast by BTC2G • © 2025', fontsize=10,
            color='white', ha='right', va='bottom', alpha=0.7)

    yticks = np.arange(0, 220001, 10000)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"${y:,}" for y in yticks], color='white', rotation=45)

    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
    ax.legend(facecolor='#1f1f1f', edgecolor='gray', labelcolor='white')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    old_files = glob.glob(os.path.join(script_dir, "btc_gold_forecast_*.png"))
    for f in old_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Failed to remove {f}: {e}")

    filename = f"btc_gold_forecast_{today_str}.png"
    filepath = os.path.join(script_dir, filename)

    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    print(f"Chart saved to: {filepath}")


if __name__ == '__main__':
    main()