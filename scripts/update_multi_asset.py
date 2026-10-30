import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


START_DATE = "2024-01-01"
TICKERS = {
    "Bitcoin": "BTC-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "SPX": "^GSPC",
}
PAIRS = {
    "btc_gold": ("Bitcoin", "Gold", "BTC / Gold"),
    "btc_silver": ("Bitcoin", "Silver", "BTC / Silver"),
    "btc_spx": ("Bitcoin", "SPX", "BTC / S&P 500"),
    "gold_spx": ("Gold", "SPX", "Gold / S&P 500"),
}
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "assets" / "multi_asset_strength.json"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def download_close(ticker: str, label: str) -> pd.Series:
    data = yf.download(
        ticker,
        start=START_DATE,
        end=utc_now().strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"Failed to download {label} ({ticker}).")
    close = data["Adj Close"] if "Adj Close" in data else data.get("Close")
    if close is None:
        raise RuntimeError(f"No close price for {label} ({ticker}).")
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        raise RuntimeError(f"Close prices were empty for {label} ({ticker}).")
    close.name = label
    return close


def _change_at_or_before(series: pd.Series, days: int) -> float | None:
    cutoff = series.index[-1] - pd.Timedelta(days=days)
    earlier = series.loc[:cutoff]
    if earlier.empty:
        return None
    previous = float(earlier.iloc[-1])
    latest = float(series.iloc[-1])
    return round((latest / previous - 1) * 100, 2) if previous > 0 else None


def build_payload(prices: pd.DataFrame, generated_at: datetime | None = None, limit: int = 365) -> dict:
    required = list(TICKERS)
    if not isinstance(prices, pd.DataFrame) or any(column not in prices for column in required):
        raise ValueError("All four source series are required.")
    aligned = prices[required].dropna().sort_index()
    if len(aligned) < 31 or not aligned.index.is_monotonic_increasing or aligned.index.has_duplicates:
        raise ValueError("At least 31 ordered common observations are required.")
    if not np.isfinite(aligned.to_numpy()).all() or (aligned <= 0).any().any():
        raise ValueError("Prices must be finite and positive.")
    generated_at = generated_at or utc_now()
    market_date = aligned.index[-1].date()
    age = (generated_at.date() - market_date).days
    if age < 0 or age > 4:
        raise ValueError(f"Latest common market date is invalid or stale ({age} days).")

    ratios = pd.DataFrame(index=aligned.index)
    pair_metadata = {}
    for key, (numerator, denominator, label) in PAIRS.items():
        ratios[key] = aligned[numerator] / aligned[denominator]
        pair_metadata[key] = {
            "label": label,
            "numerator": TICKERS[numerator],
            "denominator": TICKERS[denominator],
        }
    if not np.isfinite(ratios.to_numpy()).all() or (ratios <= 0).any().any():
        raise ValueError("Relative-strength ratios must be finite and positive.")

    current_pairs = {}
    for key in PAIRS:
        current_pairs[key] = {
            "value": round(float(ratios[key].iloc[-1]), 6),
            "change_7d_pct": _change_at_or_before(ratios[key], 7),
            "change_30d_pct": _change_at_or_before(ratios[key], 30),
        }
    btc_keys = ["btc_gold", "btc_silver", "btc_spx"]
    comparable = [key for key in btc_keys if current_pairs[key]["change_30d_pct"] is not None]
    strongest = max(comparable, key=lambda key: current_pairs[key]["change_30d_pct"]) if comparable else None

    rows = []
    for date, row in ratios.tail(limit).iterrows():
        rows.append({"date": date.strftime("%Y-%m-%d"), **{
            key: round(float(row[key]), 6) for key in PAIRS
        }})
    return {
        "source": "yfinance",
        "tickers": TICKERS,
        "price_basis": "Latest completed common daily close; no forward-filled quotes",
        "updated_for_market_date": market_date.isoformat(),
        "generated_at_utc": generated_at.isoformat(),
        "pairs": pair_metadata,
        "current": {"pairs": current_pairs, "strongest_btc_30d": strongest},
        "series": rows,
        "disclaimer": "Relative price ratios, not exchange rates or investment recommendations.",
    }


def main() -> int:
    try:
        prices = pd.concat(
            [download_close(ticker, label) for label, ticker in TICKERS.items()],
            axis=1,
            sort=False,
        )
        payload = build_payload(prices)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = OUTPUT_PATH.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, OUTPUT_PATH)
        print(f"Multi-asset strength saved to: {OUTPUT_PATH}")
        return 0
    except Exception as exc:
        print(f"Multi-asset update failed: {exc}", file=sys.stderr)
        if OUTPUT_PATH.exists():
            print(f"Keeping existing data: {OUTPUT_PATH}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
