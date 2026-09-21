"""SQLite persistence layer for BLACKLEDGER."""
import sqlite3
import pandas as pd
from pathlib import Path
from config import DB_PATH
from logger import get_logger

log = get_logger(__name__)


class DatabaseManager:
    """Thin wrapper around SQLite for storing transactions, predictions, and metadata."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    # ── Schema ─────────────────────────────────────────────────────

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file TEXT,
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id INTEGER,
                    model_name TEXT,
                    fraud_probability REAL,
                    predicted_class INTEGER,
                    risk_score REAL,
                    anomaly_flag INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # ── model_metrics: migrate legacy schema if needed ─────
            # The table must have a column literally named "model" to
            # match the DataFrame produced by ModelEvaluator.evaluate_all().
            if self._table_exists(conn, "model_metrics"):
                existing_cols = self._column_names(conn, "model_metrics")
                if "model" not in existing_cols:
                    log.warning(
                        "Legacy model_metrics schema detected (%s). "
                        "Dropping and recreating with correct columns.",
                        existing_cols,
                    )
                    conn.execute("DROP TABLE model_metrics")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT,
                    precision REAL,
                    recall REAL,
                    f1 REAL,
                    roc_auc REAL,
                    pr_auc REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

        log.info("Database schema initialised at %s", self.db_path)

    @staticmethod
    def _table_exists(conn, table: str) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        return cur.fetchone() is not None

    @staticmethod
    def _column_names(conn, table: str):
        cur = conn.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cur.fetchall()]

    # ── Generic helpers ────────────────────────────────────────────

    def save_dataframe(self, df: pd.DataFrame, table: str,
                       if_exists: str = "replace"):
        with self._connect() as conn:
            df.to_sql(table, conn, if_exists=if_exists, index=False)
        log.info("Saved %d rows to table '%s'", len(df), table)

    def load_dataframe(self, query: str) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(query, conn)

    # ── Metrics ────────────────────────────────────────────────────

    def save_metrics(self, metrics_df: pd.DataFrame):
        """Persist model metrics. Aligns column names with the table schema."""
        df = metrics_df.copy()

        # The evaluator emits "model"; the table also uses "model".
        # If a different caller supplies "model_name", rename it so the
        # insert succeeds.
        if "model_name" in df.columns and "model" not in df.columns:
            df = df.rename(columns={"model_name": "model"})

        # Keep only columns the table actually has.
        with self._connect() as conn:
            valid_cols = self._column_names(conn, "model_metrics")
        keep = [c for c in df.columns if c in valid_cols]
        df = df[keep]

        with self._connect() as conn:
            df.to_sql("model_metrics", conn, if_exists="append", index=False)
        log.info("Saved %d metric rows to model_metrics", len(df))

    def close(self):
        log.info("Database connection closed.")