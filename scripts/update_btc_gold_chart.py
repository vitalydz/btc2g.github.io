import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


START_DATE = "2016-01-01"
FUTURE_YEARS = 5
FUTURE_DAYS = FUTURE_YEARS * 365
TICKERS = {
    "Gold": "GC=F",
    "Bitcoin": "BTC-USD",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
CHART_PATH = ASSETS_DIR / "btc_gold_forecast.png"
META_PATH = ASSETS_DIR / "btc_gold_forecast_meta.json"
SIGNAL_PATH = ASSETS_DIR / "btc_signal.json"
HISTORY_PATH = ASSETS_DIR / "btc_gold_history.json"


def utc_today() -> datetime:
    return datetime.now(timezone.utc)


def download_data(ticker: str, label: str) -> pd.Series:
    data = yf.download(
        ticker,
        start=START_DATE,
        end=utc_today().strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"Failed to download data for {label} ({ticker}).")

    if "Adj Close" in data:
        close = data["Adj Close"]
    elif "Close" in data:
        close = data["Close"]
    else:
        raise RuntimeError(f"No close price data found for {label} ({ticker}).")

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.dropna()
    if close.empty:
        raise RuntimeError(f"Downloaded close prices were empty for {label} ({ticker}).")

    close.name = label
    return close


def date_to_num(dates) -> np.ndarray:
    return np.array([date.to_pydatetime().date().toordinal() for date in dates])


def validate_market_date(df: pd.DataFrame) -> None:
    """Allow weekends/short holiday closures, but never silently publish stale data."""
    market_date = df.index.max().date()
    age = (utc_today().date() - market_date).days
    if age < 0 or age > 4:
        raise RuntimeError(f"Latest common market date {market_date} is invalid or stale ({age} days).")


def forecast_polyfit(dates, prices, future_days: int, degree: int = 2):
    x = date_to_num(dates)
    y = prices.values.astype(float)
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)

    last_day = dates[-1].to_pydatetime().date()
    future_dates = [last_day + timedelta(days=i) for i in range(1, future_days + 1)]
    x_future = np.array([day.toordinal() for day in future_dates])

    return future_dates, poly(x_future)


def build_signal(df: pd.DataFrame) -> dict:
    """Create a cautious model-based BTC signal from the BTC/Gold ratio.

    This MVP rule compares the latest BTC/Gold ratio with its 365-day rolling
    average. It is intentionally simple and should be reviewed before being
    used as a paid signal product.
    """
    prices = df[["Bitcoin", "Gold"]].dropna().sort_index()
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise RuntimeError("BTC/Gold prices must be finite and positive.")
    ratio = prices["Bitcoin"] / prices["Gold"]
    if not np.isfinite(ratio.to_numpy()).all() or (ratio <= 0).any():
        raise RuntimeError("BTC/Gold ratios must be finite and positive.")
    if len(ratio) < 365:
        raise RuntimeError("Not enough BTC/Gold ratio data to build a signal.")

    latest_ratio = float(ratio.iloc[-1])
    published_ratio = round(latest_ratio, 3)
    if published_ratio <= 0:
        raise RuntimeError("BTC/Gold ratio is too small to publish at three decimals.")
    rolling_window = ratio.tail(365)
    rolling_mean = float(rolling_window.mean())
    rolling_std = float(rolling_window.std())
    if not np.isfinite(rolling_mean) or not np.isfinite(rolling_std):
        raise RuntimeError("BTC/Gold rolling statistics are invalid.")

    if rolling_std <= 0:
        signal = "HOLD"
        confidence = 0.5
        note = "BTC appears neutral compared to gold based on the model"
    else:
        z_score = (latest_ratio - rolling_mean) / rolling_std
        confidence = round(min(0.95, 0.5 + min(abs(z_score), 2.0) * 0.2), 2)

        if z_score <= -0.5:
            signal = "BUY"
            note = "BTC appears undervalued compared to gold based on the model"
        elif z_score >= 0.5:
            signal = "SELL"
            note = "BTC appears overvalued compared to gold based on the model"
        else:
            signal = "HOLD"
            note = "BTC appears neutral compared to gold based on the model"

    seven_day_cutoff = ratio.index[-1] - pd.Timedelta(days=7)
    earlier = ratio.loc[:seven_day_cutoff]
    ratio_7d_ago = float(earlier.iloc[-1]) if not earlier.empty else None
    change_7d_pct = (
        round((latest_ratio / ratio_7d_ago - 1) * 100, 2)
        if ratio_7d_ago and np.isfinite(ratio_7d_ago) and ratio_7d_ago > 0
        else None
    )

    return {
        "signal": signal,
        "confidence": confidence,
        "ratio": published_ratio,
        "ratio_7d_ago": round(ratio_7d_ago, 3) if ratio_7d_ago else None,
        "ratio_change_7d_pct": change_7d_pct,
        "btc_price_usd": round(float(prices["Bitcoin"].iloc[-1]), 2),
        "gold_price_usd": round(float(prices["Gold"].iloc[-1]), 2),
        "last_updated": ratio.index[-1].strftime("%Y-%m-%d"),
        "note": note,
    }


def build_history(df: pd.DataFrame, limit: int = 730) -> dict:
    prices = df[["Bitcoin", "Gold"]].dropna().sort_index()
    ratio = prices["Bitcoin"] / prices["Gold"]
    rolling_mean = ratio.rolling(365, min_periods=365).mean()
    rolling_std = ratio.rolling(365, min_periods=365).std()
    rows = []
    for date, value in ratio.tail(limit).items():
        mean = rolling_mean.loc[date]
        std = rolling_std.loc[date]
        score = None
        if np.isfinite(mean) and np.isfinite(std):
            z_score = 0.0 if std <= 0 else (value - mean) / std
            score = round(min(0.95, 0.5 + min(abs(z_score), 2.0) * 0.2) * 100)
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "ratio": round(float(value), 4),
            "btc_usd": round(float(prices.loc[date, "Bitcoin"]), 2),
            "gold_usd": round(float(prices.loc[date, "Gold"]), 2),
            "model_score": score,
        })
    return {
        "source": "yfinance",
        "tickers": TICKERS,
        "units": {"ratio": "troy ounces per BTC", "btc_usd": "USD", "gold_usd": "USD per troy ounce", "model_score": "0-100 model score"},
        "updated_for_market_date": rows[-1]["date"],
        "series": rows,
    }


def build_chart() -> None:
    gold_series = download_data(TICKERS["Gold"], "Gold")
    btc_series = download_data(TICKERS["Bitcoin"], "Bitcoin")

    df = pd.concat([gold_series, btc_series], axis=1, sort=False).dropna()
    df.columns = ["Gold", "Bitcoin"]
    if df.empty:
        raise RuntimeError("Combined BTC/Gold dataset is empty.")
    validate_market_date(df)
    signal = build_signal(df)
    history = build_history(df)

    gold_future_dates, gold_forecast = forecast_polyfit(df.index, df["Gold"], FUTURE_DAYS)
    btc_future_dates, btc_forecast = forecast_polyfit(df.index, df["Bitcoin"], FUTURE_DAYS)
    today_str = utc_today().strftime("%Y-%m-%d")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#2c2c2c")
    ax.set_facecolor("#2c2c2c")

    ax.plot(df.index, df["Gold"], label="Gold (actual)", color="gold")
    ax.plot(df.index, df["Bitcoin"], label="Bitcoin (actual)", color="orange")
    ax.plot(gold_future_dates, gold_forecast, "--", label="Gold (forecast)", color="darkgoldenrod")
    ax.plot(btc_future_dates, btc_forecast, "--", label="Bitcoin (forecast)", color="red")

    ax.set_title(
        f"Bitcoin vs Gold Forecast\nFrom 2016 to {today_str}, with projection to 2030",
        fontsize=14,
        color="white",
    )
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price (USD)", color="white")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 220000)

    ax.text(
        df.index[-1],
        2000,
        "Forecast by BTC2G • © 2025",
        fontsize=10,
        color="white",
        ha="right",
        va="bottom",
        alpha=0.7,
    )

    yticks = np.arange(0, 220001, 10000)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"${y:,}" for y in yticks], color="white", rotation=45)

    ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
    ax.legend(facecolor="#1f1f1f", edgecolor="gray", labelcolor="white")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    temp_chart_path = CHART_PATH.with_suffix(".tmp.png")

    plt.tight_layout()
    plt.savefig(temp_chart_path, dpi=200)
    plt.close(fig)

    os.replace(temp_chart_path, CHART_PATH)
    SIGNAL_PATH.write_text(json.dumps(signal, indent=2) + "\n", encoding="utf-8")
    HISTORY_PATH.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    META_PATH.write_text(
        json.dumps(
            {
                "last_updated_utc": utc_today().isoformat(),
                "status": "ok",
                "market_date": signal["last_updated"],
                "price_basis": "Latest completed daily close shared by BTC and gold; no forward-filled quotes",
                "source": "yfinance",
                "tickers": TICKERS,
                "chart": str(CHART_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
                "signal": str(SIGNAL_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
                "history": str(HISTORY_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Chart saved to: {CHART_PATH}")
    print(f"Signal saved to: {SIGNAL_PATH}")
    print(f"History saved to: {HISTORY_PATH}")


def main() -> int:
    try:
        build_chart()
        return 0
    except Exception as exc:
        print(f"Chart update failed: {exc}", file=sys.stderr)
        if CHART_PATH.exists():
            print(f"Keeping existing chart: {CHART_PATH}", file=sys.stderr)
            if SIGNAL_PATH.exists():
                print(f"Keeping existing signal: {SIGNAL_PATH}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
