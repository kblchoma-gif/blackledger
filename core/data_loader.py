"""CSV import, profiling, and dataset inspection."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from logger import get_logger

log = get_logger(__name__)

class DataLoader:
    """Loads CSV files with chunking support for large datasets."""

    @staticmethod
    def load_csv(path: Path, nrows: Optional[int] = None,
                 chunksize: Optional[int] = None) -> pd.DataFrame:
        """Load a CSV, optionally in chunks."""
        log.info("Loading CSV: %s", path)
        if chunksize:
            chunks = []
            for chunk in pd.read_csv(path, chunksize=chunksize,
                                     low_memory=False):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(path, nrows=nrows, low_memory=False)
        log.info("Loaded shape=%s", df.shape)
        return df

    @staticmethod
    def profile(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a profiling dictionary for the dataset."""
        profile = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
            "n_duplicates": int(df.duplicated().sum()),
            "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_cols": df.select_dtypes(
                include=["object", "category"]).columns.tolist(),
            "datetime_cols": df.select_dtypes(
                include=["datetime64"]).columns.tolist(),
        }
        return profile

    @staticmethod
    def detect_target_column(df: pd.DataFrame) -> Optional[str]:
        """Heuristically identify the fraud label column."""
        candidates = ["class", "isfraud", "is_fraud", "fraud", "label",
                      "target", "fraudulent"]
        for col in df.columns:
            if col.lower().strip() in candidates:
                return col
        # Binary column with only 0/1
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() == 2 and set(df[col].dropna().unique()) <= {0, 1}:
                return col
        return None