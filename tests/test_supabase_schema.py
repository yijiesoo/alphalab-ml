"""Tests for Phase 3C: Supabase schema and migration."""

import json
import pytest
from pathlib import Path
from datetime import datetime

from alphalab_ml.supabase_schema import (
    MLBacktestRunsSchema,
    MLScoresSchema,
    get_ml_schema_sql,
    backup_ml_runs_to_file,
    restore_ml_runs_from_file,
    get_ml_run_stats,
)


def test_ml_backtest_runs_schema():
    """Test schema definition has correct SQL."""
    sql = MLBacktestRunsSchema.CREATE_TABLE_SQL
    
    assert "ml_backtest_runs" in sql
    assert "run_id" in sql
    assert "model_version" in sql
    assert "rebalance_date" in sql
    assert "metrics_json" in sql
    assert "portfolio_json" in sql


def test_ml_scores_schema():
    """Test scores schema definition."""
    sql = MLScoresSchema.CREATE_TABLE_SQL
    
    assert "ml_scores" in sql
    assert "score_date" in sql
    assert "ticker" in sql
    assert "score" in sql


def test_get_ml_schema_sql():
    """Test complete schema SQL generation."""
    sql = get_ml_schema_sql()
    
    # Should contain both table definitions
    assert "ml_backtest_runs" in sql
    assert "ml_scores" in sql
    
    # Should contain index creation
    assert "CREATE INDEX" in sql
    
    # Should contain constraints
    assert "CONSTRAINT" in sql


def test_backup_ml_runs_to_file_no_supabase(tmp_path):
    """Test backup gracefully handles no Supabase."""
    result = backup_ml_runs_to_file(None, str(tmp_path / "backup.json"))
    
    assert result is False


def test_backup_ml_runs_to_file_creates_file(tmp_path, monkeypatch):
    """Test backup creates valid JSON file."""
    backup_file = tmp_path / "backup.json"
    
    # Mock supabase client
    class MockSupabase:
        class TableAPI:
            def select(self, *args):
                return self
            def execute(self):
                self.data = [
                    {
                        "run_id": "test_001",
                        "model_version": "1.0",
                        "rebalance_date": "2026-03-18",
                        "metrics_json": {"ic": 0.05},
                        "portfolio_json": {"leverage": 0.7},
                        "warning": None,
                    }
                ]
                return self
        
        def table(self, name):
            return self.TableAPI()
    
    mock_client = MockSupabase()
    
    result = backup_ml_runs_to_file(mock_client, str(backup_file))
    
    # File should be created
    assert backup_file.exists()
    
    # File should contain valid JSON
    with open(backup_file, "r") as f:
        data = json.load(f)
    
    assert "backup_date" in data
    assert "total_runs" in data
    assert "runs" in data
    assert len(data["runs"]) == 1
    assert data["runs"][0]["run_id"] == "test_001"


def test_restore_ml_runs_from_file_creates_backup(tmp_path):
    """Test restore function works with backup file."""
    # Create backup file
    backup_file = tmp_path / "backup.json"
    backup_data = {
        "backup_date": datetime.now().isoformat(),
        "total_runs": 1,
        "runs": [
            {
                "run_id": "restored_001",
                "model_version": "1.0",
                "rebalance_date": "2026-03-18",
                "metrics_json": {"ic": 0.05},
                "portfolio_json": {"leverage": 0.7},
            }
        ],
    }
    
    with open(backup_file, "w") as f:
        json.dump(backup_data, f)
    
    # Mock supabase
    class MockSupabase:
        class TableAPI:
            def __init__(self):
                self.inserted = []
            
            def insert(self, data, ignore_duplicates=False):
                self.inserted.append(data)
                return self
            
            def execute(self):
                return self
        
        def table(self, name):
            return self.TableAPI()
    
    mock_client = MockSupabase()
    
    result = restore_ml_runs_from_file(mock_client, str(backup_file))
    
    assert result is True


def test_restore_ml_runs_no_supabase(tmp_path):
    """Test restore handles no Supabase gracefully."""
    backup_file = tmp_path / "backup.json"
    
    result = restore_ml_runs_from_file(None, str(backup_file))
    
    assert result is False


def test_get_ml_run_stats():
    """Test stats aggregation from runs."""
    # Mock supabase
    class MockSupabase:
        class TableAPI:
            def select(self, *args):
                return self
            
            def order(self, *args, **kwargs):
                return self
            
            def limit(self, n):
                return self
            
            def execute(self):
                self.data = [
                    {
                        "run_id": "run_001",
                        "metrics_json": {
                            "ic": 0.05,
                            "sharpe": 1.2,
                            "hit_rate": 0.55,
                        },
                    },
                    {
                        "run_id": "run_002",
                        "metrics_json": {
                            "ic": 0.03,
                            "sharpe": 0.8,
                            "hit_rate": 0.51,
                        },
                    },
                ]
                return self
        
        def table(self, name):
            return self.TableAPI()
    
    mock_client = MockSupabase()
    
    stats = get_ml_run_stats(mock_client)
    
    assert stats["total_runs"] == 2
    assert "avg_ic" in stats
    assert stats["avg_ic"] == pytest.approx(0.04, abs=0.01)
    assert "best_sharpe" in stats
    assert stats["best_sharpe"] == 1.2


def test_get_ml_run_stats_no_supabase():
    """Test stats handles no Supabase."""
    stats = get_ml_run_stats(None)
    
    assert stats == {}


def test_get_ml_run_stats_empty_runs():
    """Test stats with no runs."""
    class MockSupabase:
        class TableAPI:
            def select(self, *args):
                return self
            
            def order(self, *args, **kwargs):
                return self
            
            def limit(self, n):
                return self
            
            def execute(self):
                self.data = []
                return self
        
        def table(self, name):
            return self.TableAPI()
    
    mock_client = MockSupabase()
    stats = get_ml_run_stats(mock_client)
    
    assert stats == {}
