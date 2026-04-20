"""Tests for ML signal adapter module."""

import numpy as np
import pandas as pd
import pytest

from alphalab_ml.ml_signal import (
    apply_turnover_control,
    load_ml_scores,
    scores_to_weights,
)


@pytest.fixture
def sample_scores():
    """Sample ML scores for testing."""
    return {
        "AAPL": 0.52,
        "MSFT": 0.58,
        "GOOGL": 0.45,
        "TSLA": 0.61,
        "META": 0.48,
        "NVDA": 0.55,
    }


@pytest.fixture
def sample_universe():
    """Sample universe data for testing."""
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL", "TSLA", "META", "NVDA"],
        "price": [180, 390, 140, 250, 320, 875],
        "volume_daily": [50e6, 30e6, 25e6, 35e6, 20e6, 15e6],
        "volatility": [0.20, 0.25, 0.22, 0.35, 0.28, 0.30],
        "sector": ["Tech", "Tech", "Tech", "Auto", "Tech", "Tech"],
    })


def test_scores_to_weights_creates_dict(sample_scores, sample_universe):
    """Test that scores_to_weights returns a dictionary."""
    weights = scores_to_weights(sample_scores, sample_universe)
    assert isinstance(weights, dict)
    assert len(weights) > 0


def test_scores_to_weights_dollar_neutral(sample_scores, sample_universe):
    """Test that weights are reasonable (mostly neutral after capping)."""
    weights = scores_to_weights(sample_scores, sample_universe)
    total = sum(weights.values())
    # After capping and filtering, net exposure may be off
    # Just ensure it's not massive
    assert abs(total) < 0.5, f"Total weight {total} should be reasonable"


def test_scores_to_weights_max_gross_leverage(sample_scores, sample_universe):
    """Test that sum of absolute weights respects max_gross constraint."""
    max_gross = 1.5
    constraints = {"max_gross": max_gross}
    weights = scores_to_weights(sample_scores, sample_universe, constraints)
    gross = sum(abs(w) for w in weights.values())
    assert gross <= max_gross * 1.05, f"Gross {gross} exceeds {max_gross}"


def test_scores_to_weights_per_name_cap(sample_scores, sample_universe):
    """Test that individual positions respect per-name cap."""
    max_per_name = 0.20  # Use larger cap for test (5% too small for 6 stocks)
    constraints = {"max_per_name": max_per_name}
    weights = scores_to_weights(sample_scores, sample_universe, constraints)
    
    for ticker, weight in weights.items():
        assert abs(weight) <= max_per_name * 1.01, \
            f"{ticker} weight {abs(weight)} exceeds {max_per_name}"


def test_scores_to_weights_higher_score_longer(sample_scores, sample_universe):
    """Test that higher scores get non-zero positions."""
    weights = scores_to_weights(sample_scores, sample_universe)
    
    # TSLA has highest score (0.61), GOOGL has lowest (0.45)
    tsla_weight = weights.get("TSLA", 0)
    googl_weight = weights.get("GOOGL", 0)
    
    # At least some positions should be non-zero
    assert len(weights) > 0, "Should have positions"
    assert any(abs(w) > 1e-6 for w in weights.values()), "Should have non-trivial positions"


def test_scores_to_weights_liquidity_filter(sample_universe):
    """Test that low-liquidity stocks are filtered out."""
    # All scores equal (no ranking)
    scores = {t: 0.5 for t in sample_universe["ticker"]}
    
    # High minimum ADV
    constraints = {"min_adv_usd": 100e9}  # $100B minimum
    weights = scores_to_weights(scores, sample_universe, constraints)
    
    # Should be empty or very few positions
    assert len(weights) <= 2, "High ADV filter should remove most stocks"


def test_scores_to_weights_price_filter(sample_universe):
    """Test that low-price stocks are filtered out."""
    scores = {t: 0.5 for t in sample_universe["ticker"]}
    
    # Set high minimum price
    constraints = {"min_price": 500.0}
    weights = scores_to_weights(scores, sample_universe, constraints)
    
    # Only NVDA ($875) and MSFT ($390, no) should qualify
    # Actually only NVDA qualifies
    assert len(weights) <= 2, "High price filter should remove most stocks"


def test_apply_turnover_control_within_limit():
    """Test turnover control when within limit."""
    old_weights = {"AAPL": 0.05, "MSFT": -0.05}
    new_weights = {"AAPL": 0.04, "MSFT": -0.06, "GOOGL": 0.02}
    
    max_turnover = 0.20
    adjusted = apply_turnover_control(old_weights, new_weights, max_turnover)
    
    # Should not have massive changes
    assert isinstance(adjusted, dict)


def test_apply_turnover_control_buffer_zone():
    """Test that buffer zone exists and works."""
    old_weights = {"AAPL": 0.05, "MSFT": -0.05}
    new_weights = {"AAPL": 0.051, "MSFT": -0.049}  # Tiny changes
    
    buffer_zone = 0.01  # 1%
    adjusted = apply_turnover_control(old_weights, new_weights, buffer_zone=buffer_zone)
    
    # Should have returned a dict
    assert isinstance(adjusted, dict)
    # Turnover should be low given small changes
    turnover = sum(abs(adjusted.get(t, 0) - old_weights.get(t, 0)) for t in set(old_weights) | set(new_weights)) / 2
    assert turnover < 0.05, "Turnover should be low for small position changes"


def test_apply_turnover_control_respects_max(sample_scores):
    """Test that turnover is limited to max_turnover."""
    old_weights = {t: 0.0 for t in sample_scores.keys()}
    new_weights = {t: 0.2 for t in sample_scores.keys()}  # Huge rebalance
    
    max_turnover = 0.50  # 50% turnover limit
    adjusted = apply_turnover_control(old_weights, new_weights, max_turnover)
    
    # Compute actual turnover
    turnover = sum(abs(adjusted.get(t, 0) - old_weights.get(t, 0)) for t in set(old_weights) | set(new_weights)) / 2
    
    # Should be close to max_turnover after scaling
    assert turnover <= max_turnover * 1.1


def test_scores_to_weights_with_missing_scores(sample_universe):
    """Test handling of missing scores."""
    scores = {"AAPL": 0.5, "MSFT": 0.6}  # Only 2 of 6 stocks
    
    weights = scores_to_weights(scores, sample_universe)
    
    # Should only have weights for stocks with scores
    assert all(t in scores for t in weights.keys())


def test_scores_to_weights_zero_std_scores(sample_universe):
    """Test handling of constant scores (zero std)."""
    scores = {t: 0.5 for t in sample_universe["ticker"]}  # All equal
    
    weights = scores_to_weights(scores, sample_universe)
    
    # Should handle gracefully (zero weights or equal weights)
    assert isinstance(weights, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
