"""alphalab_ml – cross-sectional factor ML package."""

# Phase 3A: Flask API
from alphalab_ml.flask_api import (
    get_latest_ml_metrics,
    get_all_ml_backtests,
    get_ml_scores_for_ticker,
)

# Phase 3C: Supabase integration (optional)
try:
    from alphalab_ml.supabase_schema import (
        create_ml_tables,
        backup_ml_runs_to_file,
        restore_ml_runs_from_file,
        migrate_json_runs_to_supabase,
        get_ml_run_stats,
    )
except ImportError:
    pass  # Supabase not required

__all__ = [
    # Flask API queries
    "get_latest_ml_metrics",
    "get_all_ml_backtests",
    "get_ml_scores_for_ticker",
    # Supabase utilities
    "create_ml_tables",
    "backup_ml_runs_to_file",
    "restore_ml_runs_from_file",
    "migrate_json_runs_to_supabase",
    "get_ml_run_stats",
]
