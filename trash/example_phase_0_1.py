"""Phase 0 + Phase 1 example: Load scores, build portfolio, run backtest."""

from pathlib import Path
import pandas as pd
from alphalab_ml.ml_signal import load_ml_scores, ingest_latest_artifact
from alphalab_ml.backtest_runner import (
    create_ml_portfolio,
    compute_oos_metrics,
    save_backtest_run,
    format_metrics_for_api,
)


def example_load_and_build_portfolio():
    """Example: Load ML scores and build portfolio."""
    
    # Step 1: Load latest artifact (manifest + scores + model)
    print("\n=== Phase 0 + 1: Load & Build ===\n")
    artifact = ingest_latest_artifact()
    print(f"Artifact version: {artifact['version']}")
    print(f"Scores shape: {artifact['scores_df'].shape}")
    
    # Step 2: Load scores for latest date
    latest_date = artifact['scores_df']['date'].max()
    scores = load_ml_scores(str(latest_date))
    print(f"Loaded {len(scores)} scores for {latest_date}")
    
    # Step 3: Create universe DataFrame (normally from market data)
    # For demo, we'll create a minimal universe
    universe_data = {
        "ticker": list(scores.keys())[:10],  # First 10 tickers
        "price": [100 + i * 10 for i in range(10)],
        "volume_daily": [50e6] * 10,
        "volatility": [0.20] * 10,
        "sector": ["Tech", "Finance", "Healthcare"] * 3 + ["Tech"],
    }
    universe_df = pd.DataFrame(universe_data)
    
    # Step 4: Filter scores to only tickers in universe
    scores_filtered = {t: scores[t] for t in universe_df["ticker"] if t in scores}
    
    # Step 5: Create portfolio with constraints
    constraints = {
        "long_pct": 0.30,
        "short_pct": 0.30,
        "max_gross": 1.5,
        "max_per_name": 0.10,
    }
    
    portfolio = create_ml_portfolio(scores_filtered, universe_df, constraints)
    print(f"\nPortfolio created:")
    print(f"  - Positions: {portfolio['coverage']}")
    print(f"  - Long exposure: {portfolio['long_exposure']:.3f}")
    print(f"  - Short exposure: {portfolio['short_exposure']:.3f}")
    print(f"  - Gross leverage: {portfolio['gross_leverage']:.3f}")
    
    # Step 6: Compute OOS metrics
    metrics = compute_oos_metrics(artifact['scores_df'])
    print(f"\nOOS Metrics:")
    print(f"  - IC: {metrics['ic']:.4f}")
    print(f"  - Hit rate: {metrics['hit_rate']:.1%}")
    print(f"  - Sharpe: {metrics['sharpe']:.2f}")
    print(f"  - Max DD: {metrics['max_dd']:.2%}")
    
    # Step 7: Save backtest run
    run = save_backtest_run(
        run_id="demo_run_001",
        model_version=artifact["version"],
        rebalance_date=str(latest_date),
        portfolio=portfolio,
        metrics=metrics,
    )
    print(f"\nBacktest run saved: {run}")
    
    # Step 8: Format for API response
    backtest_run = {
        "run_id": "demo_run_001",
        "model_version": artifact["version"],
        "rebalance_date": str(latest_date),
        "portfolio": portfolio,
        "metrics": metrics,
    }
    
    api_response = format_metrics_for_api(backtest_run)
    print(f"\nAPI Response (/api/latest-metrics):")
    import json
    print(json.dumps(api_response, indent=2))
    
    return portfolio, metrics, api_response


if __name__ == "__main__":
    try:
        portfolio, metrics, api_response = example_load_and_build_portfolio()
        print("\n✅ Example completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
