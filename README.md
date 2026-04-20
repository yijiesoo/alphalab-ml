# AlphaLab ML - Cross-Sectional Factor ML

A machine learning pipeline that **ranks stocks by relative strength** using cross-sectional factors.

## What It Does

**Input:** S&P 500 universe (last 5 years of historical data)  
**Process:** Train Ridge regression model on cross-sectional factors (momentum, reversal, volatility, moving averages)  
**Output:** ML scores for each stock (-1 to +1 scale) predicting **next rebalance period outperformance**

### Use Case
- Generate **ranked stock signals** (e.g., NVDA is top, TSM is bottom)
- Build **long/short portfolio** based on rankings
- Rebalance on schedule (daily/weekly/monthly)
- Measure predictive power via Information Coefficient (IC)

### What It Does NOT Do
- ❌ Predict absolute returns or market direction
- ❌ Provide price targets
- ❌ Replace risk management (diversification required)
- ❌ Guarantee profits (backtests show historical performance only)

## Repository Layout

```
alphalab_ml/        # Python package
  config.py         # YAML config loader
  data.py           # Universe loader + yfinance price downloader
  features.py       # Cross-sectional factor features (mom, reversal, vol, MA ratio)
  labels.py         # Forward-return label builder (no look-ahead)
  dataset.py        # Merge features + labels into tidy ML dataset
  model.py          # Walk-forward Ridge training + scoring helpers
  supabase_io.py    # Artifact upload/download via Supabase Storage
  pipeline.py       # End-to-end pipeline orchestration
configs/
  factor_ml.yaml    # Horizon, split, model hyper-parameters
data/universe/
  sp500.csv         # Committed universe (ticker list)
scripts/
  run_pipeline.py   # CLI entry point
```

## How to Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in:
#   SUPABASE_URL=
#   SUPABASE_SERVICE_ROLE_KEY=
#   SUPABASE_STORAGE_BUCKET=alphalab-ml-artifacts
```

> **Note:** The Supabase storage bucket should be set to **private**. The pipeline
> uses the service role key to upload artifacts; do **not** expose that key in
> client-side code or commit it to the repository.

### 3. Run the pipeline

```bash
python scripts/run_pipeline.py
```

This will:
1. Download S&P 500 price data via `yfinance` (cached locally in `data/cache/`)
2. Build cross-sectional features (momentum, reversal, volatility, MA ratios)
3. Generate forward-return labels (21-trading-day horizon, no look-ahead bias)
4. Train Ridge regression model using walk-forward backtesting
5. Score the latest snapshot → `reports/latest_scores.csv`
6. Store backtest metrics → `backtest_runs/` (JSON format)
7. Upload model artifacts to Supabase Storage (optional)

**Runtime:** ~2-5 minutes, ~500MB memory (S&P 500, 5 years data)

### 4. Optional: custom config

```bash
python scripts/run_pipeline.py path/to/my_config.yaml
```

Edit `configs/factor_ml.yaml` to customize:
- Training/validation/test split
- Ridge alpha parameter
- Rebalance frequency
- Feature selection

## Output Files

| File | Purpose |
|------|---------|
| `reports/latest_scores.csv` | Current stock rankings (-1 to +1) |
| `backtest_runs/*.json` | Backtest metrics (IC, Sharpe, hit rate) |
| `artifacts/ridge_model.joblib` | Trained Ridge model |
| `artifacts/manifest.json` | Model metadata (version, training window) |

## Flask API Integration

The package exports 3 query functions for dashboard integration:

```python
from alphalab_ml import (
    get_latest_ml_metrics,      # Latest backtest run metrics
    get_all_ml_backtests,       # Historical backtest runs
    get_ml_scores_for_ticker    # Individual stock score + rank
)
```

See `/flask_app/static/ml_metrics_widget.js` for dashboard widget example.

## Testing

```bash
# Run all tests (51 tests across 4 phases)
pytest tests/ -v

# Run specific test
pytest tests/test_flask_api.py -v
```

**All tests passing:** ✅ 51/51

## Architecture

### Phase 0: ML Signal Adapter
- Cross-sectional factor computation (momentum, reversal, volatility, MA ratio)
- Ranked weight generation with constraints (dollar neutral, max leverage)
- Turnover control

### Phase 1: Backtest Runner
- Walk-forward validation (no look-ahead bias)
- Ridge regression training
- Performance metrics (IC, Sharpe, hit rate, max drawdown)

### Phase 2: Leakage Checker
- Data leakage detection (future knowledge, scaler fit)
- Universe consistency validation
- Audit reports

### Phase 3: UI Integration
- **3A:** Flask API endpoints for dashboard
- **3B:** ML metrics widget (JavaScript)
- **3C:** Supabase integration (optional persistent storage)