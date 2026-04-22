# alphalab-ml

Machine learning model package for **alphalab**.

This repository contains the training/inference code used to generate ML signals/scores that can be optionally consumed by the main `alphalab` application.

## What this repo is for

- Feature engineering for financial time series / factor signals
- Model training and evaluation
- Producing prediction outputs / scores for downstream consumption
- (Optional) Packaging/export so `alphalab` can import or call it

## Requirements

- Python 3.11+ (recommended)
- pip or uv
- (Optional) GPU acceleration depending on your model stack

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use a `pyproject.toml` instead:

```bash
pip install -e .[dev]
```

## Quick start

### 1) Train a model
If your repo provides a training script, run something like:

```bash
python -m alphalab_ml.train
```

Or:

```bash
python scripts/train.py
```

### 2) Run inference / generate scores

```bash
python -m alphalab_ml.predict --ticker NVDA
```

Or:

```bash
python scripts/predict.py --input data/features.parquet --output outputs/scores.csv
```

> Update the commands above to match the actual module/script names in this repo.

## Project layout (suggested)

A typical structure looks like:

- `alphalab_ml/` — Python package
- `scripts/` — CLI entry points for training/inference
- `notebooks/` — experiments (optional)
- `data/` — local-only data (should be gitignored)
- `outputs/` — generated artifacts (should be gitignored)
- `tests/` — unit tests

## Integration with `alphalab`

The main `alphalab` app may treat this repo as an **optional dependency**.

Common integration options:

1. **Editable install (local dev)**

```bash
pip install -e ../alphalab-ml
```

2. **Package dependency**
- Publish to a package index, or reference via Git URL in `requirements.txt`

3. **Service mode**
- Run a lightweight API in this repo (FastAPI/Flask) and have `alphalab` call it

## Development

### Linting / formatting
If you use `ruff`:

```bash
ruff check .
ruff format .
```

### Running tests
If you use `pytest`:

```bash
pytest -q
```

## Reproducibility notes

- Set random seeds for numpy/torch/sklearn where relevant
- Log dataset versions and feature definitions
- Save model artifacts with metadata (date, commit SHA, config)

## License

MIT

## Related repos

- `yijiesoo/alphalab` — main application / backtester
- `yijiesoo/alphalab-ml` — this repo (ML model)