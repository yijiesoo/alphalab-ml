"""Download and cache OHLCV price data via yfinance."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

_CACHE_DIR = Path("data/cache")


def load_universe(csv_path: str | Path) -> list[str]:
    """Return the list of tickers from the universe CSV."""
    df = pd.read_csv(csv_path)
    return df["ticker"].dropna().str.strip().tolist()


def fetch_prices(
    tickers: Sequence[str],
    start: str = "2000-01-01",
    end: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Download adjusted close prices for *tickers*.

    Parameters
    ----------
    tickers:
        List of ticker symbols.
    start:
        Start date (inclusive) in ``YYYY-MM-DD`` format.
    end:
        End date (exclusive).  Defaults to today.
    cache:
        When ``True`` save a Parquet file under ``data/cache/`` and reload
        from disk on subsequent calls (keyed by sorted ticker list + start/end).

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with a ``DatetimeIndex`` and one column per ticker
        containing adjusted close prices.
    """
    key = "_".join(sorted(tickers)) + f"_{start}_{end or 'today'}"
    key_hash = hashlib.md5(key.encode()).hexdigest()
    cache_file = _CACHE_DIR / f"{key_hash}.parquet"

    if cache and cache_file.exists():
        log.info("Loading prices from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    log.info("Downloading prices for %d tickers via yfinance …", len(tickers))
    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    prices: pd.DataFrame = raw["Close"].copy()
    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)

    if cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_file)
        log.info("Cached prices to %s", cache_file)

    return prices
