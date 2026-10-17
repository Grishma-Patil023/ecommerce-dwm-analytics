"""
etl/db.py — Universal Database adapter supporting SQLite (offline default) and PostgreSQL.
"""
import sqlite3
from typing import Any, List, Dict, Optional, Tuple
import pandas as pd
from .config import DB_TYPE, SQLITE_PATH, SUPABASE_DB_URL

def get_connection():
    """Returns database connection based on configured backend."""
    if DB_TYPE == "postgres" and SUPABASE_DB_URL:
        try:
            import psycopg
            return psycopg.connect(SUPABASE_DB_URL, autocommit=True)
        except ImportError:
            raise RuntimeError("psycopg not installed. Run: pip install 'psycopg[binary]'")
    
    # Default to SQLite
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def execute_script(sql_script: str) -> None:
    """Executes a multi-statement SQL script."""
    conn = get_connection()
    try:
        if hasattr(conn, "executescript"):
            # SQLite
            conn.executescript(sql_script)
            conn.commit()
        else:
            # PostgreSQL
            with conn.cursor() as cur:
                cur.execute(sql_script)
    finally:
        conn.close()

def query_df(sql: str, params: Optional[Tuple] = None) -> pd.DataFrame:
    """Execute SQL query and return results as pandas DataFrame."""
    conn = get_connection()
    try:
        if params:
            return pd.read_sql_query(sql, conn, params=params)
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()

def execute_query(sql: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
    """Execute SQL, return list of dicts."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        if cur.description:
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            return [dict(zip(cols, row)) for row in rows]
        if hasattr(conn, "commit"):
            conn.commit()
        return []
    finally:
        conn.close()

def execute_batch(sql: str, data: List[Tuple]) -> int:
    """Execute batch insert."""
    if not data:
        return 0
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.executemany(sql, data)
        if hasattr(conn, "commit"):
            conn.commit()
        return len(data)
    finally:
        conn.close()
