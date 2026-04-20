"""Phase 3C: Supabase database schema and migration utilities.

This module provides:
  1. Schema definitions for storing ML backtest runs
  2. Migration scripts to create tables
  3. Backup/restore functions for data persistence

Uses Supabase REST API (no yfinance calls).
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Any

log = logging.getLogger(__name__)


# =====================================================================
# Schema Definitions
# =====================================================================

class MLBacktestRunsSchema:
    """SQL schema for ml_backtest_runs table."""
    
    TABLE_NAME = "ml_backtest_runs"
    
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS ml_backtest_runs (
        id BIGSERIAL PRIMARY KEY,
        run_id TEXT NOT NULL UNIQUE,
        model_version TEXT NOT NULL,
        rebalance_date DATE NOT NULL,
        metrics_json JSONB NOT NULL,
        portfolio_json JSONB NOT NULL,
        warning TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        
        -- Indexes for common queries
        CONSTRAINT valid_ic CHECK ((metrics_json->>'ic')::FLOAT BETWEEN -1 AND 1),
        CONSTRAINT valid_hit_rate CHECK ((metrics_json->>'hit_rate')::FLOAT BETWEEN 0 AND 1)
    );
    
    CREATE INDEX IF NOT EXISTS idx_ml_runs_created_at 
        ON ml_backtest_runs(created_at DESC);
    
    CREATE INDEX IF NOT EXISTS idx_ml_runs_rebalance_date 
        ON ml_backtest_runs(rebalance_date DESC);
    
    CREATE INDEX IF NOT EXISTS idx_ml_runs_model_version 
        ON ml_backtest_runs(model_version);
    """
    
    DROP_TABLE_SQL = "DROP TABLE IF EXISTS ml_backtest_runs CASCADE;"


class MLScoresSchema:
    """SQL schema for ml_scores table (store score history)."""
    
    TABLE_NAME = "ml_scores"
    
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS ml_scores (
        id BIGSERIAL PRIMARY KEY,
        score_date DATE NOT NULL,
        ticker TEXT NOT NULL,
        score FLOAT NOT NULL,
        rank INT,
        percentile FLOAT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        
        -- Composite index for querying by date and ticker
        UNIQUE(score_date, ticker)
    );
    
    CREATE INDEX IF NOT EXISTS idx_ml_scores_date 
        ON ml_scores(score_date DESC);
    
    CREATE INDEX IF NOT EXISTS idx_ml_scores_ticker 
        ON ml_scores(ticker);
    
    CREATE INDEX IF NOT EXISTS idx_ml_scores_date_ticker 
        ON ml_scores(score_date DESC, ticker);
    """
    
    DROP_TABLE_SQL = "DROP TABLE IF EXISTS ml_scores CASCADE;"


# =====================================================================
# Migration Functions
# =====================================================================


def create_ml_tables(supabase_client) -> bool:
    """Create ML backtest tables in Supabase.
    
    Parameters
    ----------
    supabase_client : supabase.Client
        Initialized Supabase client.
    
    Returns
    -------
    bool
        True if tables created successfully, False otherwise.
    """
    try:
        if not supabase_client:
            log.warning("⚠️ Supabase not configured")
            return False
        
        # Execute SQL to create tables
        # Note: This requires Supabase SQL Editor or direct RPC call
        # For now, we'll use REST API to check if tables exist
        
        try:
            # Try to query the table to see if it exists
            response = supabase_client.table(
                MLBacktestRunsSchema.TABLE_NAME
            ).select("*").limit(1).execute()
            
            log.info("✅ ml_backtest_runs table already exists")
        except Exception as e:
            if "does not exist" in str(e):
                log.warning("⚠️ ml_backtest_runs table does not exist. "
                           "Please create it manually in Supabase SQL Editor:")
                log.warning(MLBacktestRunsSchema.CREATE_TABLE_SQL)
                return False
        
        try:
            # Check ml_scores table
            response = supabase_client.table(
                MLScoresSchema.TABLE_NAME
            ).select("*").limit(1).execute()
            
            log.info("✅ ml_scores table already exists")
        except Exception as e:
            if "does not exist" in str(e):
                log.warning("⚠️ ml_scores table does not exist. "
                           "Please create it manually in Supabase SQL Editor:")
                log.warning(MLScoresSchema.CREATE_TABLE_SQL)
                return False
        
        return True
    
    except Exception as e:
        log.error("❌ Error creating tables: %s", e)
        return False


def get_ml_schema_sql() -> str:
    """Get the complete SQL schema for ML tables.
    
    Returns
    -------
    str
        SQL script that can be run in Supabase SQL Editor.
    """
    return f"""
-- ML Backtest Runs Schema
{MLBacktestRunsSchema.CREATE_TABLE_SQL}

-- ML Scores History Schema
{MLScoresSchema.CREATE_TABLE_SQL}

-- Optional: Create materialized view for recent runs
CREATE MATERIALIZED VIEW IF NOT EXISTS vw_ml_recent_runs AS
SELECT 
    run_id,
    model_version,
    rebalance_date,
    metrics_json,
    portfolio_json,
    warning,
    created_at
FROM ml_backtest_runs
ORDER BY created_at DESC
LIMIT 100;

CREATE INDEX IF NOT EXISTS idx_vw_ml_recent_created_at 
    ON vw_ml_recent_runs(created_at DESC);
"""


# =====================================================================
# Backup & Restore Functions
# =====================================================================


def backup_ml_runs_to_file(
    supabase_client,
    output_file: str = "ml_backtest_runs_backup.json",
) -> bool:
    """Backup all ML backtest runs to a JSON file.
    
    Parameters
    ----------
    supabase_client : supabase.Client
        Initialized Supabase client.
    output_file : str
        Path to write backup file.
    
    Returns
    -------
    bool
        True if backup succeeded, False otherwise.
    """
    try:
        if not supabase_client:
            log.warning("⚠️ Supabase not configured")
            return False
        
        # Fetch all runs
        response = supabase_client.table(
            MLBacktestRunsSchema.TABLE_NAME
        ).select("*").execute()
        
        runs = response.data or []
        
        backup_data = {
            "backup_date": datetime.now().isoformat(),
            "total_runs": len(runs),
            "runs": runs,
        }
        
        # Write to file
        with open(output_file, "w") as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        log.info("✅ Backed up %d ML runs to %s", len(runs), output_file)
        return True
    
    except Exception as e:
        log.error("❌ Error backing up ML runs: %s", e)
        return False


def restore_ml_runs_from_file(
    supabase_client,
    backup_file: str,
) -> bool:
    """Restore ML backtest runs from a JSON backup file.
    
    Parameters
    ----------
    supabase_client : supabase.Client
        Initialized Supabase client.
    backup_file : str
        Path to backup file.
    
    Returns
    -------
    bool
        True if restore succeeded, False otherwise.
    """
    try:
        if not supabase_client:
            log.warning("⚠️ Supabase not configured")
            return False
        
        # Load backup
        with open(backup_file, "r") as f:
            backup_data = json.load(f)
        
        runs = backup_data.get("runs", [])
        
        # Insert runs (ignoring duplicates)
        for run in runs:
            try:
                response = supabase_client.table(
                    MLBacktestRunsSchema.TABLE_NAME
                ).insert(run, ignore_duplicates=True).execute()
                
            except Exception as e:
                log.warning("⚠️ Failed to restore run %s: %s", 
                           run.get("run_id"), e)
                continue
        
        log.info("✅ Restored %d ML runs from %s", len(runs), backup_file)
        return True
    
    except Exception as e:
        log.error("❌ Error restoring ML runs: %s", e)
        return False


def migrate_json_runs_to_supabase(
    supabase_client,
    json_runs_dir: str = "backtest_runs",
) -> int:
    """Migrate local JSON backtest runs to Supabase.
    
    Parameters
    ----------
    supabase_client : supabase.Client
        Initialized Supabase client.
    json_runs_dir : str
        Directory containing JSON backtest runs.
    
    Returns
    -------
    int
        Number of runs successfully migrated.
    """
    try:
        if not supabase_client:
            log.warning("⚠️ Supabase not configured")
            return 0
        
        from pathlib import Path
        
        runs_dir = Path(json_runs_dir)
        if not runs_dir.exists():
            log.warning("⚠️ Directory %s does not exist", json_runs_dir)
            return 0
        
        # Load all JSON files
        migrated_count = 0
        
        for json_file in runs_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    run_data = json.load(f)
                
                # Insert into Supabase
                response = supabase_client.table(
                    MLBacktestRunsSchema.TABLE_NAME
                ).insert(run_data, ignore_duplicates=True).execute()
                
                migrated_count += 1
                log.info("✅ Migrated %s", json_file.name)
                
            except Exception as e:
                log.warning("⚠️ Failed to migrate %s: %s", json_file.name, e)
                continue
        
        log.info("✅ Migrated %d runs to Supabase", migrated_count)
        return migrated_count
    
    except Exception as e:
        log.error("❌ Error migrating runs: %s", e)
        return 0


# =====================================================================
# Query Functions (For Dashboard)
# =====================================================================


def get_recent_ml_runs(
    supabase_client,
    limit: int = 20,
) -> List[dict]:
    """Get recent ML backtest runs from Supabase.
    
    Parameters
    ----------
    supabase_client : supabase.Client
        Initialized Supabase client.
    limit : int
        Maximum number of runs to return.
    
    Returns
    -------
    list
        List of backtest run dictionaries.
    """
    try:
        if not supabase_client:
            return []
        
        response = supabase_client.table(
            MLBacktestRunsSchema.TABLE_NAME
        ).select("*").order("created_at", desc=True).limit(limit).execute()
        
        return response.data or []
    
    except Exception as e:
        log.warning("⚠️ Error querying recent runs: %s", e)
        return []


def get_ml_run_stats(supabase_client) -> dict:
    """Get aggregate statistics about ML runs.
    
    Parameters
    ----------
    supabase_client : supabase.Client
        Initialized Supabase client.
    
    Returns
    -------
    dict
        Statistics like total_runs, avg_ic, best_sharpe, etc.
    """
    try:
        if not supabase_client:
            return {}
        
        runs = get_recent_ml_runs(supabase_client, limit=1000)
        
        if not runs:
            return {}
        
        ics = []
        sharpes = []
        hit_rates = []
        
        for run in runs:
            metrics = run.get("metrics_json", {})
            
            if isinstance(metrics, str):
                metrics = json.loads(metrics)
            
            if "ic" in metrics:
                ics.append(metrics["ic"])
            if "sharpe" in metrics:
                sharpes.append(metrics["sharpe"])
            if "hit_rate" in metrics:
                hit_rates.append(metrics["hit_rate"])
        
        stats = {
            "total_runs": len(runs),
            "avg_ic": sum(ics) / len(ics) if ics else None,
            "best_ic": max(ics) if ics else None,
            "worst_ic": min(ics) if ics else None,
            "avg_sharpe": sum(sharpes) / len(sharpes) if sharpes else None,
            "best_sharpe": max(sharpes) if sharpes else None,
            "avg_hit_rate": sum(hit_rates) / len(hit_rates) if hit_rates else None,
        }
        
        log.info("📊 ML run stats: %d runs, avg IC=%.4f", 
                len(runs), stats.get("avg_ic", 0))
        
        return stats
    
    except Exception as e:
        log.warning("⚠️ Error computing stats: %s", e)
        return {}
