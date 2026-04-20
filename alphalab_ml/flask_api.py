"""Phase 3A: Flask API integration for ML backtest metrics.

This module provides query functions that power:
  - GET /api/latest-metrics
  - GET /api/all-backtests
  - GET /api/ml-scores/<ticker>

It loads backtest runs from JSON and (optionally) Supabase.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import pandas as pd

log = logging.getLogger(__name__)

# Get the root directory of the alphalab-ml package
_PKG_ROOT = Path(__file__).parent.parent
_BACKTEST_DIR = _PKG_ROOT / "backtest_runs"
_REPORTS_DIR = _PKG_ROOT / "reports"


# =====================================================================
# Query Functions (No yfinance calls - uses only local data)
# =====================================================================


def get_latest_ml_metrics() -> dict[str, Any]:
    """Get latest ML backtest metrics for API response.
    
    Loads from backtest_runs/ folder (JSON-based, no yfinance calls).
    
    Returns
    -------
    dict
        Response for GET /api/latest-metrics
        Structure:
        {
            "status": "success" or "error",
            "model_version": str,
            "as_of_date": str (YYYY-MM-DD),
            "metrics": {
                "ic": float [-1, 1],
                "hit_rate": float [0, 1],
                "sharpe": float,
                "max_drawdown": float,
                "turnover": float,
            },
            "coverage": {
                "universe_size": int,
                "valid_scores": int,
            },
            "portfolio": {
                "long_exposure": float,
                "short_exposure": float,
                "gross_leverage": float,
            },
            "warning": str or null,
            "timestamp": str (ISO 8601),
        }
    """
    try:
        _BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        
        # Find latest backtest run
        runs = sorted(
            _BACKTEST_DIR.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        
        if not runs:
            return {
                "status": "no_data",
                "message": "No backtest runs available yet",
                "timestamp": datetime.now().isoformat(),
            }
        
        # Load latest run
        with open(runs[0], "r") as f:
            run = json.load(f)
        
        log.info("✅ Loaded latest backtest: %s", runs[0].name)
        
        portfolio = run.get("portfolio", {})
        metrics = run.get("metrics", {})
        
        # Extract coverage from portfolio (it's a scalar in portfolio.coverage)
        portfolio_coverage = portfolio.get("coverage", 0)
        
        return {
            "status": "success",
            "model_version": run.get("model_version", "unknown"),
            "as_of_date": run.get("rebalance_date", "unknown"),
            "metrics": {
                "ic": metrics.get("ic", 0.0),
                "hit_rate": metrics.get("hit_rate", 0.0),
                "sharpe": metrics.get("sharpe", 0.0),
                "max_drawdown": metrics.get("max_dd", 0.0),
                "turnover": metrics.get("turnover", 0.0),
            },
            "portfolio": {
                "long_exposure": portfolio.get("long_exposure", 0.0),
                "short_exposure": portfolio.get("short_exposure", 0.0),
                "gross_leverage": portfolio.get("gross_leverage", 0.0),
            },
            "coverage": {
                "universe_size": 500,  # S&P 500
                "valid_scores": portfolio_coverage,
            },
            "warning": run.get("warning"),
            "timestamp": run.get("timestamp"),
        }
    
    except Exception as e:
        log.error("❌ Error loading latest metrics: %s", e)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def get_all_ml_backtests(limit: int = 50) -> dict[str, Any]:
    """Get historical backtest runs for API response.
    
    Loads from backtest_runs/ folder (JSON-based, no yfinance calls).
    
    Parameters
    ----------
    limit : int
        Maximum number of runs to return (default 50).
    
    Returns
    -------
    dict
        Response for GET /api/all-backtests?limit=50
        Structure:
        {
            "status": "success",
            "backtests": [
                {
                    "run_id": str,
                    "model_version": str,
                    "rebalance_date": str,
                    "metrics": {...},
                    "timestamp": str,
                },
                ...
            ],
            "total": int,
            "limit": int,
        }
    """
    try:
        _BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        
        # Find all backtest runs, sorted by date desc
        runs = sorted(
            _BACKTEST_DIR.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )[:limit]
        
        backtests = []
        for run_file in runs:
            try:
                with open(run_file, "r") as f:
                    run = json.load(f)
                backtests.append({
                    "run_id": run.get("run_id", run_file.stem),
                    "model_version": run.get("model_version", "unknown"),
                    "rebalance_date": run.get("rebalance_date", "unknown"),
                    "metrics": run.get("metrics", {}),
                    "timestamp": run.get("timestamp"),
                })
            except Exception as e:
                log.warning("⚠️ Failed to load %s: %s", run_file.name, e)
                continue
        
        log.info("✅ Loaded %d backtest runs", len(backtests))
        
        return {
            "status": "success",
            "backtests": backtests,
            "total": len(backtests),
            "limit": limit,
            "timestamp": datetime.now().isoformat(),
        }
    
    except Exception as e:
        log.error("❌ Error loading backtests: %s", e)
        return {
            "status": "error",
            "message": str(e),
            "backtests": [],
            "total": 0,
            "limit": limit,
            "timestamp": datetime.now().isoformat(),
        }


def get_ml_scores_for_ticker(ticker: str) -> dict[str, Any]:
    """Get latest ML scores for a specific ticker.
    
    Loads from latest_scores.csv (no yfinance calls).
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL").
    
    Returns
    -------
    dict
        Response for GET /api/ml-scores/<ticker>
        Structure:
        {
            "status": "success" or "error",
            "ticker": str,
            "score": float or null,
            "rank": int or null,
            "percentile": float or null,
            "as_of_date": str,
            "timestamp": str,
        }
    """
    try:
        scores_file = _REPORTS_DIR / "latest_scores.csv"
        
        if not scores_file.exists():
            return {
                "status": "no_data",
                "ticker": ticker.upper(),
                "message": "No scores file available yet",
                "timestamp": datetime.now().isoformat(),
            }
        
        # Load scores
        df = pd.read_csv(scores_file)
        df["ticker"] = df["ticker"].str.upper()
        
        # Find ticker
        row = df[df["ticker"] == ticker.upper()]
        if row.empty:
            return {
                "status": "not_found",
                "ticker": ticker.upper(),
                "message": f"Ticker {ticker.upper()} not in current universe",
                "timestamp": datetime.now().isoformat(),
            }
        
        row = row.iloc[0]
        
        # Compute rank and percentile
        n_scores = len(df)
        rank = (df["score"] > row["score"]).sum() + 1
        percentile = rank / n_scores * 100
        
        log.info("✅ Found score for %s: %.4f (rank %d/%d)", 
                 ticker.upper(), row["score"], rank, n_scores)
        
        return {
            "status": "success",
            "ticker": ticker.upper(),
            "score": float(row["score"]),
            "rank": int(rank),
            "percentile": float(percentile),
            "as_of_date": row.get("date", "unknown"),
            "timestamp": datetime.now().isoformat(),
        }
    
    except Exception as e:
        log.error("❌ Error loading score for %s: %s", ticker, e)
        return {
            "status": "error",
            "ticker": ticker.upper(),
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
        }


# =====================================================================
# Supabase Integration (Phase 3C - Optional)
# =====================================================================


def save_backtest_to_supabase(
    supabase_client,
    run_id: str,
    model_version: str,
    rebalance_date: str,
    metrics: dict,
    portfolio: dict,
    warning: Optional[str] = None,
) -> bool:
    """Save backtest run to Supabase for persistence.
    
    This is OPTIONAL and only called if Supabase is configured.
    Falls back gracefully if Supabase is unavailable.
    
    Parameters
    ----------
    supabase_client : supabase.Client
        Initialized Supabase client.
    run_id : str
        Unique run ID.
    model_version : str
        Model version string.
    rebalance_date : str
        Date of rebalance (YYYY-MM-DD).
    metrics : dict
        Backtest metrics (ic, hit_rate, sharpe, etc.).
    portfolio : dict
        Portfolio stats (long_exp, short_exp, gross_lev).
    warning : str, optional
        Warning message if any.
    
    Returns
    -------
    bool
        True if saved successfully, False otherwise.
    """
    try:
        if not supabase_client:
            log.warning("⚠️ Supabase not configured, skipping persistence")
            return False
        
        data = {
            "run_id": run_id,
            "model_version": model_version,
            "rebalance_date": rebalance_date,
            "metrics_json": json.dumps(metrics),
            "portfolio_json": json.dumps(portfolio),
            "warning": warning,
            "created_at": datetime.now().isoformat(),
        }
        
        response = supabase_client.table("ml_backtest_runs").insert(data).execute()
        log.info("✅ Saved backtest to Supabase: %s", run_id)
        return True
    
    except Exception as e:
        log.warning("⚠️ Failed to save backtest to Supabase: %s (falling back to JSON)", e)
        return False


def load_backtest_from_supabase(
    supabase_client,
    run_id: str,
) -> Optional[dict]:
    """Load backtest run from Supabase.
    
    Parameters
    ----------
    supabase_client : supabase.Client
        Initialized Supabase client.
    run_id : str
        Unique run ID to load.
    
    Returns
    -------
    dict or None
        Backtest run data if found, None otherwise.
    """
    try:
        if not supabase_client:
            return None
        
        response = supabase_client.table("ml_backtest_runs").select("*").eq("run_id", run_id).execute()
        
        if not response.data:
            return None
        
        row = response.data[0]
        return {
            "run_id": row.get("run_id"),
            "model_version": row.get("model_version"),
            "rebalance_date": row.get("rebalance_date"),
            "metrics": json.loads(row.get("metrics_json", "{}")),
            "portfolio": json.loads(row.get("portfolio_json", "{}")),
            "warning": row.get("warning"),
            "created_at": row.get("created_at"),
        }
    
    except Exception as e:
        log.warning("⚠️ Failed to load backtest from Supabase: %s", e)
        return None
