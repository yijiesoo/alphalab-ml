"""Merge features and labels into a clean ML dataset."""

from __future__ import annotations

import pandas as pd

from .features import build_features
from .labels import forward_return


FEATURE_COLS = ["mom", "str_rev", "vol", "ma_ratio"]


def build_dataset(prices: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Join cross-sectional features with forward-return labels.

    Parameters
    ----------
    prices:
        Wide adjusted-close price DataFrame (DatetimeIndex × ticker columns).
    horizon:
        Forward-return horizon in trading days.

    Returns
    -------
    pd.DataFrame
        Tidy dataset with columns ``date``, ``ticker``, all feature columns,
        and ``label``.  Rows containing any NaN are dropped.
    """
    features_df = build_features(prices)
    labels_df = forward_return(prices, horizon=horizon)

    df = features_df.merge(labels_df, on=["date", "ticker"], how="inner")
    df = df.dropna(subset=FEATURE_COLS + ["label"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df
