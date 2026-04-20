# Problem Definition

This project focuses on cross-sectional factor machine learning, aiming to predict the forward 21-trading-day return of selected financial assets. The target variable is the forward return over a horizon of 21 days, using a no look-ahead policy to ensure robustness in predictions.

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
2. Build cross-sectional features and forward-return labels
3. Train a Ridge regression model using a walk-forward split
4. Score the latest snapshot and write `reports/latest_scores.csv`
5. Upload the model and score files to the configured Supabase bucket

### 4. Optional: custom config

```bash
python scripts/run_pipeline.py path/to/my_config.yaml
```

Artifacts are uploaded to Supabase Storage via a service role key for easier management and access.