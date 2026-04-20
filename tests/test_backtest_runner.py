"""Tests for Phase 1 backtest runner."""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from alphalab_ml.backtest_runner import (
    compute_oos_metrics,
    create_ml_portfolio,
    save_backtest_run,
    load_backtest_runs,
    get_latest_backtest_run,
    format_metrics_for_api,
)


@pytest.fixture
def sample_oos_scores():
    """Sample OOS scores for testing."""
    return pd.DataFrame({
        "date": ["2026-01-15"] * 5,
        "ticker": ["AAPL", "MSFT", "GOOGL", "TSLA", "META"],
        "score": [0.52, 0.58, 0.45, 0.61, 0.48],
        "actual": [0.02, 0.015, -0.01, 0.03, -0.005],
    })


@pytest.fixture
def sample_portfolio():
    """Sample portfolio data."""
    return {
        "weights": {"AAPL": 0.05, "MSFT": 0.06, "GOOGL": -0.04},
        "long_exposure": 0.11,
        "short_exposure": 0.04,
        "gross_leverage": 0.15,
        "coverage": 3,
    }


@pytest.fixture
def sample_metrics():
    """Sample metrics."""
    return {
        "ic": 0.052,
        "hit_rate": 0.515,
        "sharpe": 1.18,
        "max_dd": -0.16,
        "turnover": 0.22,
    }


@pytest.fixture
def sample_universe():
    """Sample universe data."""
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL", "TSLA", "META"],
        "price": [180, 390, 140, 250, 320],
        "volume_daily": [50e6, 30e6, 25e6, 35e6, 20e6],
        "volatility": [0.20, 0.25, 0.22, 0.35, 0.28],
        "sector": ["Tech", "Tech", "Tech", "Auto", "Tech"],
    })


def test_compute_oos_metrics_with_actual(sample_oos_scores):
    """Test OOS metrics computation with actual values."""
    metrics = compute_oos_metrics(sample_oos_scores)
    
    assert isinstance(metrics, dict)
    assert "ic" in metrics
    assert "hit_rate" in metrics
    assert "sharpe" in metrics
    assert "max_dd" in metrics
    
    # IC should be between -1 and 1
    assert -1 <= metrics["ic"] <= 1
    # Hit rate should be between 0 and 1
    assert 0 <= metrics["hit_rate"] <= 1


def test_compute_oos_metrics_without_actual():
    """Test OOS metrics when no actual values."""
    oos_scores = pd.DataFrame({
        "date": ["2026-01-15"] * 3,
        "ticker": ["AAPL", "MSFT", "GOOGL"],
        "score": [0.52, 0.58, 0.45],
    })
    
    metrics = compute_oos_metrics(oos_scores)
    
    assert metrics["ic"] == 0.0
    assert metrics["hit_rate"] == 0.0


def test_create_ml_portfolio(sample_universe):
    """Test ML portfolio creation."""
    scores = {"AAPL": 0.52, "MSFT": 0.58, "GOOGL": 0.45}
    
    portfolio = create_ml_portfolio(scores, sample_universe)
    
    assert "weights" in portfolio
    assert "long_exposure" in portfolio
    assert "short_exposure" in portfolio
    assert "gross_leverage" in portfolio
    assert "coverage" in portfolio
    
    assert len(portfolio["weights"]) > 0
    assert portfolio["coverage"] == len(portfolio["weights"])


def test_save_and_load_backtest_run(sample_portfolio, sample_metrics, tmp_path):
    """Test saving and loading backtest run."""
    import os
    os.chdir(tmp_path)
    
    run_id = "test_run_1"
    model_version = "ridge-v1"
    rebalance_date = "2026-03-18"
    
    # Save
    save_path = save_backtest_run(
        run_id=run_id,
        model_version=model_version,
        rebalance_date=rebalance_date,
        portfolio=sample_portfolio,
        metrics=sample_metrics,
    )
    
    assert save_path.exists()
    
    # Load
    runs = load_backtest_runs(limit=10)
    assert len(runs) > 0
    
    latest = runs[0]
    assert latest["run_id"] == run_id
    assert latest["model_version"] == model_version


def test_get_latest_backtest_run(sample_portfolio, sample_metrics, tmp_path):
    """Test getting latest backtest run."""
    import os
    os.chdir(tmp_path)
    
    # Save multiple runs
    for i in range(3):
        save_backtest_run(
            run_id=f"run_{i}",
            model_version="ridge-v1",
            rebalance_date="2026-03-18",
            portfolio=sample_portfolio,
            metrics=sample_metrics,
        )
    
    latest = get_latest_backtest_run()
    assert latest is not None
    assert "run_id" in latest


def test_format_metrics_for_api(sample_portfolio, sample_metrics):
    """Test formatting metrics for API response."""
    backtest_run = {
        "run_id": "test_1",
        "model_version": "ridge-v1",
        "rebalance_date": "2026-03-18",
        "portfolio": sample_portfolio,
        "metrics": sample_metrics,
        "warning": None,
    }
    
    response = format_metrics_for_api(backtest_run)
    
    assert response["status"] == "success"
    assert response["model_version"] == "ridge-v1"
    assert "metrics" in response
    assert "coverage" in response
    assert "portfolio" in response
    
    # Check metric values
    assert response["metrics"]["ic"] == sample_metrics["ic"]
    assert response["metrics"]["sharpe"] == sample_metrics["sharpe"]
    assert response["coverage"]["valid_scores"] == sample_portfolio["coverage"]


def test_format_metrics_for_api_empty():
    """Test formatting when no backtest runs."""
    response = format_metrics_for_api(None)
    
    assert response["status"] == "no_data"
    assert "message" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
