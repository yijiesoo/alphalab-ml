"""Cross-sectional factor feature engineering.

All features are *cross-sectionally ranked* (rank / n) to remove level
effects and make them comparable across tickers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally rank each row, returning values in [0, 1]."""
    return df.rank(axis=1, pct=True)


def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 month momentum (skip most-recent ``skip`` days to avoid reversal)."""
    total_return = prices.shift(skip) / prices.shift(lookback + skip) - 1
    return _cs_rank(total_return)


def short_term_reversal(prices: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Short-term reversal (1-month return, sign-flipped)."""
    ret = prices / prices.shift(lookback) - 1
    return _cs_rank(-ret)


def volatility(prices: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """Realised volatility (negative sign → low-vol is high rank)."""
    log_ret = np.log(prices / prices.shift(1))
    vol = log_ret.rolling(lookback).std() * np.sqrt(252)
    return _cs_rank(-vol)


def moving_average_ratio(prices: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Price relative to its moving average (trend signal)."""
    ratio = prices.rolling(fast).mean() / prices.rolling(slow).mean() - 1
    return _cs_rank(ratio)


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Concatenate all features into a long-form tidy DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``ticker``, ``mom``, ``str_rev``, ``vol``, ``ma_ratio``.
    """
    feature_map = {
        "mom": momentum(prices),
        "str_rev": short_term_reversal(prices),
        "vol": volatility(prices),
        "ma_ratio": moving_average_ratio(prices),
    }

    frames = []
    for name, wide_df in feature_map.items():
        long = wide_df.stack(future_stack=True).rename(name)
        frames.append(long)

    combined = pd.concat(frames, axis=1)
    combined.index.names = ["date", "ticker"]
    combined = combined.reset_index()
    return combined
