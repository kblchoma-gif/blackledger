"""Cleaning pipeline: missing values, duplicates, outliers."""
import pandas as pd
import numpy as np
from logger import get_logger

log = get_logger(__name__)

class DataCleaner:
    """Configurable cleaning pipeline."""

    @staticmethod
    def clean(df: pd.DataFrame, target_col: str,
              drop_duplicates: bool = True,
              fill_numeric: str = "median",
              fill_categorical: str = "mode") -> pd.DataFrame:
        df = df.copy()
        original_shape = df.shape

        # 1. Drop duplicates
        if drop_duplicates:
            df = df.drop_duplicates().reset_index(drop=True)
            log.info("Dropped %d duplicates", original_shape[0] - len(df))

        # 2. Handle missing values
        for col in df.columns:
            if col == target_col:
                df = df.dropna(subset=[col])
                continue
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    if fill_numeric == "median":
                        df[col] = df[col].fillna(df[col].median())
                    elif fill_numeric == "mean":
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(0)
                else:
                    mode = df[col].mode()
                    df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "UNKNOWN")

        # 3. Drop constant columns
        const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1 and c != target_col]
        if const_cols:
            df = df.drop(columns=const_cols)
            log.info("Dropped constant columns: %s", const_cols)

        log.info("Cleaning complete: %s → %s", original_shape, df.shape)
        return df

    @staticmethod
    def outlier_summary(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
        """Return IQR-based outlier counts per numeric column."""
        rows = []
        for col in numeric_cols:
            if col not in df.columns:
                continue
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
            rows.append({"column": col, "q1": q1, "q3": q3,
                         "iqr": iqr, "lower": lo, "upper": hi,
                         "outliers": n_out,
                         "outlier_pct": round(n_out / len(df) * 100, 2)})
        return pd.DataFrame(rows)