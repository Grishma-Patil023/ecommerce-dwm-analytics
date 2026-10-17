"""
etl/config.py — Configuration and path management for E-Commerce Data Warehouse.
"""
import os
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "ecommerce_dataset_10000.csv"
SQL_DIR = BASE_DIR / "sql"
ML_DIR = BASE_DIR / "ml_results"
ML_DIR.mkdir(exist_ok=True)

def load_env(env_path: Path = None) -> Dict[str, str]:
    """Parse .env file if available."""
    if env_path is None:
        env_path = BASE_DIR / ".env"
    env_vars = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env_vars[k.strip()] = v.strip().strip("'\"")
    return env_vars

ENV = load_env()
DB_TYPE = ENV.get("DB_TYPE", "sqlite").lower()
SQLITE_PATH = BASE_DIR / ENV.get("SQLITE_DB_PATH", "ecommerce_warehouse.db")
SUPABASE_DB_URL = ENV.get("SUPABASE_DB_URL") or os.environ.get("SUPABASE_DB_URL")
BATCH_SIZE = int(ENV.get("BATCH_SIZE", 5000))
