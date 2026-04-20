"""Build forward-return labels with no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(prices: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Cross-sectionally ranked forward return over *horizon* trading days.

    The label at date *t* is computed from prices at *t* and *t+horizon*.
    This is safe to use as a training target only for rows where the full
    horizon is available (i.e., exclude the last ``horizon`` rows).

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns ``date``, ``ticker``, ``label``.
    """
    fwd_ret = prices.shift(-horizon) / prices - 1
    # Cross-sectional rank so the model learns relative ordering
    fwd_ranked = fwd_ret.rank(axis=1, pct=True)

    long = fwd_ranked.stack(future_stack=True).rename("label")
    long.index.names = ["date", "ticker"]
    long = long.reset_index()
    return long
