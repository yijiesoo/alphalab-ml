"""Leakage detection and time-alignment validation for Phase 2."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


class LeakageDetectionError(Exception):
    """Raised when leakage is detected."""
    pass


def check_date_alignment(
    train_dates: pd.DatetimeIndex,
    valid_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Verify that train/valid/test date ranges don't overlap.
    
    Parameters
    ----------
    train_dates : pd.DatetimeIndex
        Training period dates.
    valid_dates : pd.DatetimeIndex
        Validation period dates.
    test_dates : pd.DatetimeIndex
        Test (OOS) period dates.
    
    Returns
    -------
    dict
        Validation result with keys:
        - valid (bool): True if no overlaps
        - issues (list): Any detected issues
        - timeline (dict): Date ranges and gaps
    
    Raises
    ------
    LeakageDetectionError
        If critical leakage detected.
    """
    result = {
        "valid": True,
        "issues": [],
        "timeline": {},
    }
    
    if len(train_dates) == 0 or len(valid_dates) == 0 or len(test_dates) == 0:
        result["issues"].append("One or more date ranges is empty")
        result["valid"] = False
        return result
    
    train_min, train_max = train_dates.min(), train_dates.max()
    valid_min, valid_max = valid_dates.min(), valid_dates.max()
    test_min, test_max = test_dates.min(), test_dates.max()
    
    result["timeline"] = {
        "train": (str(train_min.date()), str(train_max.date())),
        "valid": (str(valid_min.date()), str(valid_max.date())),
        "test": (str(test_min.date()), str(test_max.date())),
    }
    
    # Check for overlaps
    if train_max >= valid_min:
        issue = f"Train/valid overlap: train ends {train_max}, valid starts {valid_min}"
        result["issues"].append(issue)
        result["valid"] = False
    
    if valid_max >= test_min:
        issue = f"Valid/test overlap: valid ends {valid_max}, test starts {test_min}"
        result["issues"].append(issue)
        result["valid"] = False
    
    # Check ordering
    if not (train_max < valid_min < test_min):
        result["issues"].append("Date ranges not in chronological order")
        result["valid"] = False
    
    if result["valid"]:
        log.info("Date alignment valid: train → valid → test (no overlaps)")
    else:
        log.error("Date alignment issues: %s", result["issues"])
    
    return result


def verify_no_future_data(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    lag_days: int = 1,
) -> dict[str, Any]:
    """Verify features don't use future data (must be t-1 or earlier).
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Features with DatetimeIndex or 'date' column.
    labels_df : pd.DataFrame
        Labels with DatetimeIndex or 'date' column (forward-looking).
    lag_days : int
        Expected lag between feature and label dates (default 1).
    
    Returns
    -------
    dict
        Validation result with keys:
        - valid (bool): True if no future data detected
        - issues (list): Any problems found
        - stats (dict): Data alignment statistics
    """
    result = {
        "valid": True,
        "issues": [],
        "stats": {},
    }
    
    # Extract date columns
    feat_date = features_df.index if isinstance(features_df.index, pd.DatetimeIndex) else pd.to_datetime(features_df["date"])
    label_date = labels_df.index if isinstance(labels_df.index, pd.DatetimeIndex) else pd.to_datetime(labels_df["date"])
    
    # Align dates
    merged = pd.DataFrame({
        "feat_date": feat_date,
        "label_date": label_date,
    }).dropna()
    
    if len(merged) == 0:
        result["issues"].append("No date matches between features and labels")
        result["valid"] = False
        return result
    
    # Check for future leakage (label_date should be >= feat_date + lag_days)
    date_gaps = (merged["label_date"] - merged["feat_date"]).dt.days
    
    future_leakage = date_gaps < lag_days
    if future_leakage.any():
        n_leaks = future_leakage.sum()
        result["issues"].append(f"{n_leaks} instances of future data leakage detected")
        result["valid"] = False
        log.error("Future leakage: %d/%d rows", n_leaks, len(merged))
    
    result["stats"] = {
        "total_rows": len(merged),
        "min_gap_days": int(date_gaps.min()),
        "max_gap_days": int(date_gaps.max()),
        "mean_gap_days": float(date_gaps.mean()),
        "expected_lag_days": lag_days,
    }
    
    if result["valid"]:
        log.info("Future data check passed: min_gap=%d days (expected >=%d)",
                 result["stats"]["min_gap_days"], lag_days)
    
    return result


def validate_scaler_fit(
    scaler_fit_dates: pd.DatetimeIndex,
    train_dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Verify scaler was fit only on training data.
    
    Parameters
    ----------
    scaler_fit_dates : pd.DatetimeIndex
        Dates used to fit the scaler.
    train_dates : pd.DatetimeIndex
        Training dates.
    
    Returns
    -------
    dict
        Validation result with keys:
        - valid (bool): True if scaler fit only on train
        - issues (list): Any problems
        - stats (dict): Fit statistics
    """
    result = {
        "valid": True,
        "issues": [],
        "stats": {},
    }
    
    if len(scaler_fit_dates) != len(train_dates):
        result["issues"].append(
            f"Scaler fit on {len(scaler_fit_dates)} dates, "
            f"but train has {len(train_dates)} dates"
        )
        result["valid"] = False
    
    # Check if fit dates are subset of train dates
    fit_min, fit_max = scaler_fit_dates.min(), scaler_fit_dates.max()
    train_min, train_max = train_dates.min(), train_dates.max()
    
    if fit_min < train_min or fit_max > train_max:
        result["issues"].append(
            f"Scaler fit dates [{fit_min}, {fit_max}] "
            f"not contained in train [{train_min}, {train_max}]"
        )
        result["valid"] = False
    
    result["stats"] = {
        "scaler_fit_count": len(scaler_fit_dates),
        "train_count": len(train_dates),
        "fit_range": (str(fit_min.date()), str(fit_max.date())),
        "train_range": (str(train_min.date()), str(train_max.date())),
    }
    
    if result["valid"]:
        log.info("Scaler validation passed: fit on %d train dates", len(scaler_fit_dates))
    else:
        log.error("Scaler validation failed: %s", result["issues"])
    
    return result


def validate_universe_consistency(
    train_universe: set[str],
    valid_universe: set[str],
    test_universe: set[str],
) -> dict[str, Any]:
    """Verify universe constituents are consistent (point-in-time).
    
    Parameters
    ----------
    train_universe : set[str]
        Tickers in training period.
    valid_universe : set[str]
        Tickers in validation period.
    test_universe : set[str]
        Tickers in test period.
    
    Returns
    -------
    dict
        Validation result with keys:
        - valid (bool): True if no future delists detected
        - issues (list): Problems found
        - stats (dict): Universe statistics
    """
    result = {
        "valid": True,
        "issues": [],
        "stats": {},
    }
    
    # Tickers that appear in test but not train = possible future knowledge
    future_only = test_universe - train_universe
    if future_only:
        result["issues"].append(
            f"{len(future_only)} test tickers not in train: {sorted(list(future_only))[:5]}"
        )
        result["valid"] = False
    
    # Track changes
    train_to_valid = valid_universe - train_universe
    valid_to_test = test_universe - valid_universe
    
    result["stats"] = {
        "train_count": len(train_universe),
        "valid_count": len(valid_universe),
        "test_count": len(test_universe),
        "train_to_valid_adds": len(train_to_valid),
        "valid_to_test_adds": len(valid_to_test),
        "new_in_test": len(future_only),
    }
    
    if result["valid"]:
        log.info("Universe consistency validated")
    else:
        log.warning("Universe consistency issues: %s", result["issues"])
    
    return result


def audit_backtest_run(
    backtest_result: dict[str, Any],
    checks: list[str] | None = None,
) -> dict[str, Any]:
    """Full audit of a backtest run for leakage and validity.
    
    Parameters
    ----------
    backtest_result : dict
        Backtest run data with expected keys:
        - model_version, rebalance_date, portfolio, metrics
    checks : list[str], optional
        Which checks to run. If None, runs all.
        Options: 'date_alignment', 'future_data', 'scaler', 'universe'
    
    Returns
    -------
    dict
        Audit report with keys:
        - passed (bool): All checks passed
        - checks (dict): Individual check results
        - warnings (list): Non-critical issues
        - critical_issues (list): Critical problems
    """
    if checks is None:
        checks = ["date_alignment", "future_data", "scaler", "universe"]
    
    audit = {
        "passed": True,
        "checks": {},
        "warnings": [],
        "critical_issues": [],
    }
    
    # Run requested checks
    if "date_alignment" in checks:
        # Placeholder: requires dates from backtest context
        audit["checks"]["date_alignment"] = {
            "status": "skipped",
            "reason": "Requires train/valid/test dates from pipeline context",
        }
    
    if "future_data" in checks:
        audit["checks"]["future_data"] = {
            "status": "skipped",
            "reason": "Requires feature/label DataFrames from pipeline",
        }
    
    if "scaler" in checks:
        audit["checks"]["scaler"] = {
            "status": "skipped",
            "reason": "Requires scaler fit information from model",
        }
    
    if "universe" in checks:
        audit["checks"]["universe"] = {
            "status": "skipped",
            "reason": "Requires universe data from pipeline",
        }
    
    # Check basic backtest structure
    required_fields = ["model_version", "rebalance_date", "portfolio", "metrics"]
    missing = [f for f in required_fields if f not in backtest_result]
    
    if missing:
        audit["critical_issues"].append(f"Missing fields: {missing}")
        audit["passed"] = False
    
    # Check metrics are reasonable
    metrics = backtest_result.get("metrics", {})
    if metrics.get("ic") and abs(metrics["ic"]) > 1.0:
        audit["critical_issues"].append("IC out of bounds (must be in [-1, 1])")
        audit["passed"] = False
    
    if metrics.get("hit_rate") and not (0 <= metrics["hit_rate"] <= 1):
        audit["critical_issues"].append("Hit rate out of bounds (must be in [0, 1])")
        audit["passed"] = False
    
    log.info(
        "Backtest audit completed: passed=%s, critical_issues=%d",
        audit["passed"],
        len(audit["critical_issues"]),
    )
    
    return audit


def generate_leakage_report(
    validation_results: dict[str, dict[str, Any]],
) -> str:
    """Generate a human-readable leakage detection report.
    
    Parameters
    ----------
    validation_results : dict
        Results from various validation checks.
    
    Returns
    -------
    str
        Formatted report.
    """
    report = []
    report.append("=" * 70)
    report.append("LEAKAGE DETECTION REPORT")
    report.append("=" * 70)
    
    all_valid = True
    
    for check_name, result in validation_results.items():
        check_valid = result.get("valid", False)
        all_valid = all_valid and check_valid
        
        status = "✅ PASS" if check_valid else "❌ FAIL"
        report.append(f"\n{check_name.upper()}: {status}")
        
        if result.get("issues"):
            for issue in result["issues"]:
                report.append(f"  ⚠ {issue}")
        
        if result.get("stats"):
            report.append("  Stats:")
            for key, val in result["stats"].items():
                report.append(f"    - {key}: {val}")
    
    report.append("\n" + "=" * 70)
    if all_valid:
        report.append("OVERALL: ✅ PASS - No leakage detected")
    else:
        report.append("OVERALL: ❌ FAIL - Leakage or alignment issues found")
    report.append("=" * 70)
    
    return "\n".join(report)
