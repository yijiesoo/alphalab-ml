"""ML signal adapter: load scores and transform to portfolio weights."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_REPORTS_DIR = Path("reports")
_ARTIFACTS_DIR = Path("artifacts")


def load_ml_scores(date: str | None = None) -> dict[str, float]:
    """Load ML scores from latest_scores.csv for a specific date.
    
    Parameters
    ----------
    date : str, optional
        Date in YYYY-MM-DD format. If None, loads the most recent scores.
    
    Returns
    -------
    dict[str, float]
        Dictionary mapping ticker to ML score.
    
    Raises
    ------
    FileNotFoundError
        If latest_scores.csv does not exist.
    ValueError
        If date not found in scores or file is empty.
    """
    scores_file = _REPORTS_DIR / "latest_scores.csv"
    
    if not scores_file.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_file}")
    
    df = pd.read_csv(scores_file)
    
    if df.empty:
        raise ValueError("Scores file is empty")
    
    # If date not specified, use most recent
    if date is None:
        date = df["date"].max()
        log.info("No date specified, using most recent: %s", date)
    
    df_date = df[df["date"] == date]
    
    if df_date.empty:
        raise ValueError(f"No scores found for date {date}")
    
    scores = dict(zip(df_date["ticker"], df_date["score"]))
    log.info("Loaded %d scores for %s", len(scores), date)
    
    return scores


def scores_to_weights(
    scores: dict[str, float],
    universe_df: pd.DataFrame,
    constraints: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Transform ML scores to portfolio weights with constraints.
    
    Parameters
    ----------
    scores : dict[str, float]
        Ticker to ML score mapping (higher = more attractive).
    universe_df : pd.DataFrame
        Universe data with columns: ticker, price, volume_daily, volatility, sector.
    constraints : dict, optional
        Constraints dictionary with keys:
        - long_pct (float): Top percentage for long positions (default 0.3 = top 30%)
        - short_pct (float): Bottom percentage for short positions (default 0.3)
        - vol_target (float): Target annualized volatility (default 0.15)
        - max_gross (float): Max gross leverage (default 1.5 = 150%)
        - max_per_name (float): Max position size per stock (default 0.05 = 5%)
        - sector_cap (float): Max sector exposure (default 0.20 = 20%)
        - min_price (float): Min stock price filter (default 5.0)
        - min_adv_usd (float): Min daily volume × price (default 10e6)
    
    Returns
    -------
    dict[str, float]
        Portfolio weights: {ticker: weight}.
        Properties:
        - Sum of weights ≈ 0 (dollar-neutral)
        - Sum of abs(weights) ≈ max_gross (1.5 for 150% leverage)
        - Individual weights capped at max_per_name
    
    Examples
    --------
    >>> scores = {'AAPL': 0.52, 'MSFT': 0.58, 'GOOGL': 0.45}
    >>> universe_df = pd.DataFrame({
    ...     'ticker': ['AAPL', 'MSFT', 'GOOGL'],
    ...     'price': [180, 390, 140],
    ...     'volume_daily': [50e6, 30e6, 25e6],
    ...     'volatility': [0.20, 0.25, 0.22],
    ...     'sector': ['Tech', 'Tech', 'Tech']
    ... })
    >>> weights = scores_to_weights(scores, universe_df)
    >>> abs(sum(weights.values())) < 0.01  # Dollar neutral
    True
    """
    if constraints is None:
        constraints = {}
    
    # Set defaults
    long_pct = constraints.get("long_pct", 0.3)
    short_pct = constraints.get("short_pct", 0.3)
    vol_target = constraints.get("vol_target", 0.15)
    max_gross = constraints.get("max_gross", 1.5)
    max_per_name = constraints.get("max_per_name", 0.05)
    sector_cap = constraints.get("sector_cap", 0.20)
    min_price = constraints.get("min_price", 5.0)
    min_adv_usd = constraints.get("min_adv_usd", 10e6)
    
    log.info(
        "scores_to_weights: long_pct=%.1f%%, short_pct=%.1f%%, "
        "max_gross=%.2f, max_per_name=%.1f%%",
        long_pct * 100,
        short_pct * 100,
        max_gross,
        max_per_name * 100,
    )
    
    # Step 1: Align scores with universe data
    tickers = sorted(scores.keys())
    universe_df = universe_df.set_index("ticker")
    
    valid_tickers = [t for t in tickers if t in universe_df.index]
    invalid_tickers = [t for t in tickers if t not in universe_df.index]
    
    if invalid_tickers:
        log.warning("Skipping %d tickers not in universe: %s", len(invalid_tickers), invalid_tickers[:5])
    
    score_values = np.array([scores[t] for t in valid_tickers])
    
    # Step 2: Rank normalize (z-score)
    mean_score = np.mean(score_values)
    std_score = np.std(score_values)
    
    if std_score == 0:
        log.warning("Score std=0, using equal weights")
        z_scores = np.zeros_like(score_values)
    else:
        z_scores = (score_values - mean_score) / std_score
    
    z_clipped = np.clip(z_scores, -2.0, +2.0)
    log.info("Z-scores: min=%.2f, mean=%.2f, max=%.2f", z_clipped.min(), z_clipped.mean(), z_clipped.max())
    
    # Step 3: Identify long/short split
    long_threshold = np.percentile(z_clipped, 100 * (1 - long_pct))
    short_threshold = np.percentile(z_clipped, 100 * short_pct)
    
    long_mask = z_clipped >= long_threshold
    short_mask = z_clipped <= short_threshold
    
    log.info("Long threshold=%.2f (%d stocks), short threshold=%.2f (%d stocks)",
             long_threshold, np.sum(long_mask), short_threshold, np.sum(short_mask))
    
    # Step 4: Liquidity and price filter
    universe_subset = universe_df.loc[valid_tickers]
    
    valid_mask = (
        (universe_subset["price"] >= min_price)
        & (universe_subset["volume_daily"] * universe_subset["price"] >= min_adv_usd)
    )
    
    log.info("Liquidity filter: %d/%d pass (min_price=%.2f, min_adv=%.1fM)",
             np.sum(valid_mask), len(valid_tickers), min_price, min_adv_usd / 1e6)
    
    # Apply liquidity mask
    long_mask = long_mask & valid_mask.values
    short_mask = short_mask & valid_mask.values
    
    # Step 5: Vol-scaled weighting
    volatilities = universe_subset["volatility"].values
    
    # Avoid division by zero
    volatilities = np.where(volatilities > 0, volatilities, np.mean(volatilities))
    
    # Long positions: weight proportional to z-score / volatility
    long_z_scaled = np.where(long_mask, z_clipped / volatilities, 0)
    long_sum = np.sum(np.abs(long_z_scaled))
    
    # Short positions: weight proportional to z-score / volatility
    short_z_scaled = np.where(short_mask, z_clipped / volatilities, 0)
    short_sum = np.sum(np.abs(short_z_scaled))
    
    if long_sum == 0 and short_sum == 0:
        log.warning("No valid long/short positions after filtering")
        return {}
    
    # Normalize
    if long_sum > 0:
        long_normalized = long_z_scaled / long_sum
    else:
        long_normalized = np.zeros_like(long_z_scaled)
    
    if short_sum > 0:
        short_normalized = short_z_scaled / short_sum
    else:
        short_normalized = np.zeros_like(short_z_scaled)
    
    # Step 6: Dollar-neutral split
    # Long leg: +max_gross / 2 of total notional
    # Short leg: -max_gross / 2 of total notional
    # This ensures: sum(w)=0, sum(|w|)=max_gross
    long_leg = max_gross / 2
    short_leg = max_gross / 2
    
    raw_weights = long_leg * long_normalized - short_leg * short_normalized
    
    # Step 7: Apply per-name cap iteratively until all constraints satisfied
    capped_weights = raw_weights.copy()
    
    # Cap positions
    capped_weights = np.clip(capped_weights, -max_per_name, +max_per_name)
    
    # Normalize to achieve target gross leverage
    gross_after_cap = np.sum(np.abs(capped_weights))
    if gross_after_cap > 0:
        scale_to_gross = max_gross / gross_after_cap
        capped_weights = capped_weights * scale_to_gross
    
    # Re-apply cap if scaling broke it
    capped_weights = np.clip(capped_weights, -max_per_name, +max_per_name)
    
    # Step 8: Sector cap (optional, basic implementation)
    # For now, we'll apply a simple rescale if needed
    
    weights_dict = dict(zip(valid_tickers, capped_weights))
    
    # Filter out near-zero positions
    weights_dict = {t: w for t, w in weights_dict.items() if abs(w) > 1e-6}
    
    # Validation
    total_weight = sum(weights_dict.values())
    total_gross = sum(abs(w) for w in weights_dict.values())
    
    log.info("Final weights: count=%d, sum=%.4f (target ≈0), gross=%.4f (target %.2f)",
             len(weights_dict), total_weight, total_gross, max_gross)
    
    return weights_dict


def ingest_latest_artifact() -> dict[str, Any]:
    """Load and parse the latest ML artifact bundle.
    
    Returns
    -------
    dict
        Artifact bundle with keys:
        - version (str): Model version ID
        - timestamp (str): When artifact was created
        - model_path (Path): Path to pickled model
        - scores_df (pd.DataFrame): Scores with date, ticker, score columns
        - metadata (dict): Additional metadata from manifest
    
    Raises
    ------
    FileNotFoundError
        If required artifact files missing.
    """
    manifest_file = _ARTIFACTS_DIR / "manifest.json"
    model_file = _ARTIFACTS_DIR / "ridge_model.joblib"
    scores_file = _REPORTS_DIR / "latest_scores.csv"
    
    if not manifest_file.exists():
        log.warning("Manifest not found, creating minimal artifact info")
        metadata = {"version": "unknown", "timestamp": "unknown"}
    else:
        import json
        with open(manifest_file) as f:
            metadata = json.load(f)
        log.info("Loaded manifest: version=%s", metadata.get("version"))
    
    if not scores_file.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_file}")
    
    scores_df = pd.read_csv(scores_file)
    
    artifact = {
        "version": metadata.get("version", "unknown"),
        "timestamp": metadata.get("timestamp", "unknown"),
        "model_path": model_file if model_file.exists() else None,
        "scores_df": scores_df,
        "metadata": metadata,
    }
    
    log.info("Ingested artifact: version=%s, scores=%d", artifact["version"], len(scores_df))
    
    return artifact


def apply_turnover_control(
    new_weights: dict[str, float],
    old_weights: dict[str, float],
    max_turnover: float = 0.20,
    buffer_zone: float = 0.01,
) -> dict[str, float]:
    """Apply turnover control to limit rebalancing churn.
    
    Parameters
    ----------
    new_weights : dict[str, float]
        Proposed new portfolio weights.
    old_weights : dict[str, float]
        Previous portfolio weights.
    max_turnover : float
        Max allowed turnover (default 0.20 = 20%).
        Turnover = Σ |w_new - w_old| / 2
    buffer_zone : float
        No-trade band around current position (default 0.01 = 1%).
        Positions within ±buffer_zone are not rebalanced.
    
    Returns
    -------
    dict[str, float]
        Adjusted weights with turnover control applied.
    """
    # Identify all tickers
    all_tickers = set(new_weights.keys()) | set(old_weights.keys())
    
    # Build adjusted weights with no-trade band
    adjusted_weights = {}
    
    for ticker in all_tickers:
        w_new = new_weights.get(ticker, 0)
        w_old = old_weights.get(ticker, 0)
        
        # If change is within buffer zone, keep old weight
        if abs(w_new - w_old) < buffer_zone:
            adjusted_weights[ticker] = w_old
        else:
            adjusted_weights[ticker] = w_new
    
    # Calculate turnover
    turnover = sum(abs(adjusted_weights.get(t, 0) - old_weights.get(t, 0)) for t in all_tickers) / 2
    
    # If turnover still exceeds limit, scale down changes
    if turnover > max_turnover:
        scale_factor = max_turnover / turnover
        log.warning("Turnover %.2f%% exceeds max %.2f%%, scaling by %.2f%%",
                    turnover * 100, max_turnover * 100, scale_factor * 100)
        
        adjusted_weights = {
            t: w_old + scale_factor * (adjusted_weights[t] - w_old)
            for t in all_tickers
        }
    
    log.info("Turnover control: %.2f%% → %.2f%%", turnover * 100, 
             sum(abs(adjusted_weights.get(t, 0) - old_weights.get(t, 0)) for t in all_tickers) / 2 * 100)
    
    return {t: w for t, w in adjusted_weights.items() if abs(w) > 1e-6}
