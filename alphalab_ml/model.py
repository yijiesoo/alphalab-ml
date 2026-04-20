"""Walk-forward training and scoring with Ridge regression."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .dataset import FEATURE_COLS

log = logging.getLogger(__name__)

_ARTIFACTS_DIR = Path("artifacts")


def _make_pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
    ])


def walk_forward_train(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[Pipeline, pd.DataFrame]:
    """Train with a single walk-forward fold and return model + OOS scores.

    The last ``valid_months`` of data form the out-of-sample validation set;
    the preceding ``train_months`` are the training set.

    Parameters
    ----------
    df:
        Tidy dataset from :func:`~alphalab_ml.dataset.build_dataset`.
    cfg:
        Parsed ``factor_ml.yaml`` dict.

    Returns
    -------
    model:
        Fitted sklearn Pipeline.
    oos_scores:
        DataFrame with columns ``date``, ``ticker``, ``score`` for the
        validation period.
    """
    split_cfg = cfg.get("split", {})
    train_months: int = split_cfg.get("train_months", 60)
    valid_months: int = split_cfg.get("valid_months", 12)
    alpha: float = cfg.get("model", {}).get("alpha", 1.0)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    max_date = df["date"].max()
    valid_start = max_date - pd.DateOffset(months=valid_months)
    train_start = valid_start - pd.DateOffset(months=train_months)

    train = df[(df["date"] >= train_start) & (df["date"] < valid_start)]
    valid = df[df["date"] >= valid_start]

    log.info(
        "Train: %s → %s  (%d rows) | Valid: %s → %s  (%d rows)",
        train["date"].min().date(), train["date"].max().date(), len(train),
        valid["date"].min().date(), valid["date"].max().date(), len(valid),
    )

    model = _make_pipeline(alpha=alpha)
    model.fit(train[FEATURE_COLS], train["label"])

    oos_scores = valid[["date", "ticker"]].copy()
    oos_scores["score"] = model.predict(valid[FEATURE_COLS])

    return model, oos_scores


def score_latest(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Return cross-sectional scores for the most recent date in *df*.

    Parameters
    ----------
    model:
        Fitted sklearn Pipeline.
    df:
        Tidy dataset from :func:`~alphalab_ml.dataset.build_dataset`.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``ticker``, ``score``, sorted descending by score.
    """
    latest_date = df["date"].max()
    snap = df[df["date"] == latest_date].copy()
    snap["score"] = model.predict(snap[FEATURE_COLS])
    return snap[["date", "ticker", "score"]].sort_values("score", ascending=False).reset_index(drop=True)


def save_model(model: Pipeline, name: str = "ridge_model") -> Path:
    """Persist the model to ``artifacts/<name>.joblib``."""
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _ARTIFACTS_DIR / f"{name}.joblib"
    joblib.dump(model, out_path)
    log.info("Model saved to %s", out_path)
    return out_path


def load_model(name: str = "ridge_model") -> Pipeline:
    """Load a previously saved model from ``artifacts/<name>.joblib``."""
    model_path = _ARTIFACTS_DIR / f"{name}.joblib"
    return joblib.load(model_path)
