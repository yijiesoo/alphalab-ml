"""End-to-end pipeline orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .data import fetch_prices, load_universe
from .dataset import build_dataset
from .model import save_model, score_latest, walk_forward_train
from .supabase_io import upload_file

log = logging.getLogger(__name__)

_ARTIFACTS_DIR = Path("artifacts")
_REPORTS_DIR = Path("reports")


def run(config_path: str | Path | None = None) -> None:
    """Execute the full build → train → score-latest → upload pipeline.

    Parameters
    ----------
    config_path:
        Optional path to an alternative YAML config file.
    """
    cfg: dict[str, Any] = load_config(config_path)
    log.info("Config loaded: %s", cfg)

    # ── 1. Universe & prices ──────────────────────────────────────────────────
    universe_file: str = cfg["universe"]["file"]
    tickers = load_universe(universe_file)
    log.info("Universe: %d tickers", len(tickers))

    horizon: int = cfg["horizon_days"]
    prices = fetch_prices(tickers)

    # ── 2. Feature / label dataset ────────────────────────────────────────────
    df = build_dataset(prices, horizon=horizon)
    log.info("Dataset shape: %s", df.shape)

    # ── 3. Walk-forward train ─────────────────────────────────────────────────
    model, oos_scores = walk_forward_train(df, cfg)

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    oos_path = _REPORTS_DIR / "oos_scores.csv"
    oos_scores.to_csv(oos_path, index=False)
    log.info("OOS scores saved to %s", oos_path)

    # ── 4. Score latest snapshot ──────────────────────────────────────────────
    latest = score_latest(model, df)
    latest_path = _REPORTS_DIR / "latest_scores.csv"
    latest.to_csv(latest_path, index=False)
    log.info("Latest scores:\n%s", latest.head(10).to_string(index=False))

    # ── 5. Persist model ──────────────────────────────────────────────────────
    model_path = save_model(model)

    # ── 6. Upload artifacts to Supabase ──────────────────────────────────────
    for path in [model_path, oos_path, latest_path]:
        try:
            upload_file(path)
        except (EnvironmentError, OSError, ConnectionError, TimeoutError) as exc:
            log.warning("Supabase upload skipped (%s): %s", path.name, exc, exc_info=True)

    log.info("Pipeline complete.")
