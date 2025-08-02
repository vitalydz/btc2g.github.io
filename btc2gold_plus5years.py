# btc2gold_plus5years.py - BTC and Gold graphically statistics in last 10 years and + next 5 years forecast (2030)
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


START_DATE = '2016-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
FUTURE_YEARS = 5

TICKERS = {
    'Gold': 'GC=F',
    'Bitcoin': 'BTC-USD'
}

def download_data(ticker, label):
    data = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"Download of the data was unsuccessful. Try later. {label} ({ticker})")
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

    # Прогноз на 5 лет вперёд (~1825 дней)
    future_days = 1825
    gold_future_dates, gold_forecast = forecast_polyfit(df.index, df['Gold'], future_days)
    btc_future_dates, btc_forecast = forecast_polyfit(df.index, df['Bitcoin'], future_days)

    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Gold'], label='Gold (actual)', color='gold')
    plt.plot(df.index, df['Bitcoin'], label='Bitcoin (actual)', color='orange')
    plt.plot(gold_future_dates, gold_forecast, '--', label='Gold (forecast)', color='darkgoldenrod')
    plt.plot(btc_future_dates, btc_forecast, '--', label='Bitcoin (forecast)', color='darkred')

    plt.title('Price Forecast of Gold and Bitcoin (2016–2030)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.ylim(0, 220000)
    yticks = np.arange(0, 220000, 10000)
    plt.yticks(yticks, [f"${int(y)}" for y in yticks], rotation=45)

    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    # plt.show()
    plt.savefig("btc_gold_forecast.png", dpi=200)

if __name__ == '__main__':
    main()

