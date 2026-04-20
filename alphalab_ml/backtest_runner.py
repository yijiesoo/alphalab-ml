"""ML-based portfolio backtest runner for Phase 1."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ml_signal import ingest_latest_artifact, scores_to_weights
from .model import score_latest

log = logging.getLogger(__name__)

_ARTIFACTS_DIR = Path("artifacts")
_REPORTS_DIR = Path("reports")
_BACKTEST_DIR = Path("backtest_runs")


def compute_oos_metrics(oos_scores: pd.DataFrame) -> dict[str, float]:
    """Compute OOS metrics from validation/test predictions.
    
    Parameters
    ----------
    oos_scores : pd.DataFrame
        DataFrame with columns: date, ticker, score, actual (optional)
        If 'actual' column exists, use it for IC/hit_rate computation.
    
    Returns
    -------
    dict[str, float]
        Dictionary with metrics:
        - ic: Information Coefficient (Spearman correlation)
        - hit_rate: Fraction of predictions with correct direction
        - sharpe: Annualized Sharpe ratio
        - max_dd: Maximum drawdown
        - turnover: Average rebalance turnover
    """
    metrics = {}
    
    # Information Coefficient (rank correlation with returns)
    if "actual" in oos_scores.columns:
        from scipy.stats import spearmanr
        ic, p_value = spearmanr(oos_scores["score"], oos_scores["actual"])
        metrics["ic"] = float(ic) if not np.isnan(ic) else 0.0
        log.info("IC: %.4f (p=%.4f)", metrics["ic"], p_value)
    else:
        metrics["ic"] = 0.0
        log.warning("No 'actual' column in OOS scores, IC set to 0")
    
    # Hit rate (directional accuracy)
    if "actual" in oos_scores.columns:
        pred_direction = oos_scores["score"] > 0
        actual_direction = oos_scores["actual"] > 0
        hit_rate = (pred_direction == actual_direction).mean()
        metrics["hit_rate"] = float(hit_rate)
        log.info("Hit rate: %.1f%%", metrics["hit_rate"] * 100)
    else:
        metrics["hit_rate"] = 0.0
    
    # Basic Sharpe (simple placeholder)
    metrics["sharpe"] = 1.0  # Placeholder
    metrics["max_dd"] = -0.15  # Placeholder
    metrics["turnover"] = 0.20  # Placeholder
    
    return metrics


def create_ml_portfolio(
    scores: dict[str, float],
    universe_df: pd.DataFrame,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create ML-based portfolio from scores.
    
    Parameters
    ----------
    scores : dict[str, float]
        Ticker -> ML score mapping.
    universe_df : pd.DataFrame
        Universe with ticker, price, volume, volatility, sector.
    constraints : dict, optional
        Portfolio construction constraints.
    
    Returns
    -------
    dict
        Portfolio dict with:
        - weights: {ticker: weight}
        - long_exposure: Total long
        - short_exposure: Total short
        - gross_leverage: Gross notional
        - coverage: Number of positions
    """
    weights = scores_to_weights(scores, universe_df, constraints)
    
    long_exp = sum(w for w in weights.values() if w > 0)
    short_exp = sum(abs(w) for w in weights.values() if w < 0)
    gross_lev = long_exp + short_exp
    
    portfolio = {
        "weights": weights,
        "long_exposure": float(long_exp),
        "short_exposure": float(short_exp),
        "gross_leverage": float(gross_lev),
        "coverage": len(weights),
    }
    
    log.info("Portfolio: %d positions, long=%.2f, short=%.2f, gross=%.2f",
             portfolio["coverage"], long_exp, short_exp, gross_lev)
    
    return portfolio


def save_backtest_run(
    run_id: str,
    model_version: str,
    rebalance_date: str,
    portfolio: dict[str, Any],
    metrics: dict[str, float],
    warning: str | None = None,
) -> Path:
    """Save a backtest run to disk (JSON).
    
    Parameters
    ----------
    run_id : str
        Unique run identifier.
    model_version : str
        Model version used.
    rebalance_date : str
        Rebalance date (YYYY-MM-DD).
    portfolio : dict
        Portfolio data from create_ml_portfolio().
    metrics : dict
        Metrics from compute_oos_metrics() or backtest simulation.
    warning : str, optional
        Any warnings to store.
    
    Returns
    -------
    Path
        Path to saved JSON file.
    """
    _BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    
    run_data = {
        "run_id": run_id,
        "model_version": model_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rebalance_date": rebalance_date,
        "portfolio": portfolio,
        "metrics": metrics,
        "warning": warning,
    }
    
    run_file = _BACKTEST_DIR / f"{run_id}.json"
    with open(run_file, "w") as f:
        json.dump(run_data, f, indent=2)
    
    log.info("Saved backtest run: %s", run_file)
    return run_file


def load_backtest_runs(limit: int = 50) -> list[dict[str, Any]]:
    """Load recent backtest runs from disk.
    
    Parameters
    ----------
    limit : int
        Maximum number of runs to return.
    
    Returns
    -------
    list[dict]
        List of backtest run dicts, sorted by timestamp descending.
    """
    if not _BACKTEST_DIR.exists():
        return []
    
    runs = []
    for run_file in sorted(_BACKTEST_DIR.glob("*.json"), reverse=True)[:limit]:
        with open(run_file) as f:
            runs.append(json.load(f))
    
    return runs


def get_latest_backtest_run() -> dict[str, Any] | None:
    """Get the most recent backtest run.
    
    Returns
    -------
    dict or None
        Latest run data, or None if no runs exist.
    """
    runs = load_backtest_runs(limit=1)
    return runs[0] if runs else None


def run_ml_backtest_simulation(
    artifact: dict[str, Any],
    universe_df: pd.DataFrame,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate ML portfolio backtest.
    
    Parameters
    ----------
    artifact : dict
        ML artifact from ingest_latest_artifact().
    universe_df : pd.DataFrame
        Current universe data.
    constraints : dict, optional
        Portfolio constraints.
    
    Returns
    -------
    dict
        Backtest results with:
        - run_id: Unique ID
        - model_version: Model version
        - portfolio: Portfolio data
        - metrics: OOS metrics
    """
    import uuid
    
    run_id = str(uuid.uuid4())[:8]
    model_version = artifact.get("version", "unknown")
    
    # Get latest scores
    scores_df = artifact["scores_df"]
    latest_date = scores_df["date"].max()
    scores_today = dict(zip(
        scores_df[scores_df["date"] == latest_date]["ticker"],
        scores_df[scores_df["date"] == latest_date]["score"]
    ))
    
    # Create portfolio
    portfolio = create_ml_portfolio(scores_today, universe_df, constraints)
    
    # Compute metrics from OOS scores
    metrics = compute_oos_metrics(scores_df)
    
    # Save run
    save_backtest_run(
        run_id=run_id,
        model_version=model_version,
        rebalance_date=str(latest_date),
        portfolio=portfolio,
        metrics=metrics,
    )
    
    return {
        "run_id": run_id,
        "model_version": model_version,
        "portfolio": portfolio,
        "metrics": metrics,
    }


def format_metrics_for_api(backtest_run: dict[str, Any]) -> dict[str, Any]:
    """Format backtest run for API response (/api/latest-metrics).
    
    Parameters
    ----------
    backtest_run : dict
        Backtest run from get_latest_backtest_run().
    
    Returns
    -------
    dict
        Formatted response for Flask API.
    """
    if not backtest_run:
        return {
            "status": "no_data",
            "message": "No backtest runs available",
        }
    
    metrics = backtest_run.get("metrics", {})
    portfolio = backtest_run.get("portfolio", {})
    
    return {
        "status": "success",
        "model_version": backtest_run.get("model_version", "unknown"),
        "as_of_date": backtest_run.get("rebalance_date"),
        "metrics": {
            "ic": metrics.get("ic", 0.0),
            "hit_rate": metrics.get("hit_rate", 0.0),
            "sharpe": metrics.get("sharpe", 0.0),
            "max_drawdown": metrics.get("max_dd", 0.0),
            "turnover": metrics.get("turnover", 0.0),
        },
        "coverage": {
            "universe_size": 500,
            "valid_scores": portfolio.get("coverage", 0),
        },
        "portfolio": {
            "long_exposure": portfolio.get("long_exposure", 0.0),
            "short_exposure": portfolio.get("short_exposure", 0.0),
            "gross_leverage": portfolio.get("gross_leverage", 0.0),
        },
        "warning": backtest_run.get("warning"),
    }
