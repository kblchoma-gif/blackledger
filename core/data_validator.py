"""Validation rules for imported datasets."""
import pandas as pd
from typing import Dict, List, Any
from logger import get_logger

log = get_logger(__name__)

class DataValidator:
    """Runs a suite of validation checks and returns structured findings."""

    @staticmethod
    def validate(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        issues: List[Dict[str, str]] = []
        warnings: List[Dict[str, str]] = []

        # 1. Target column presence
        if target_col not in df.columns:
            issues.append({"level": "error", "check": "target_presence",
                           "message": f"Target column '{target_col}' not found."})

        # 2. Binary target
        if target_col in df.columns:
            uniques = set(df[target_col].dropna().unique())
            if not uniques <= {0, 1}:
                issues.append({"level": "error", "check": "binary_target",
                               "message": f"Target has non-binary values: {uniques}"})
            if df[target_col].isnull().any():
                warnings.append({"level": "warning", "check": "target_nan",
                                 "message": f"Target has {df[target_col].isnull().sum()} nulls."})

        # 3. Class imbalance report
        if target_col in df.columns:
            counts = df[target_col].value_counts()
            ratio = counts.max() / max(counts.min(), 1)
            if ratio > 10:
                warnings.append({"level": "warning", "check": "imbalance",
                                 "message": f"Severe imbalance — ratio {ratio:.1f}:1"})

        # 4. Missing values
        for col in df.columns:
            miss = df[col].isnull().mean()
            if miss > 0.50:
                warnings.append({"level": "warning", "check": "missing",
                                 "message": f"'{col}' is {miss:.0%} missing."})

        # 5. Duplicate rows
        dups = df.duplicated().sum()
        if dups > 0:
            warnings.append({"level": "warning", "check": "duplicates",
                             "message": f"{dups} duplicate rows detected."})

        # 6. Constant columns
        for col in df.columns:
            if df[col].nunique(dropna=False) <= 1:
                warnings.append({"level": "warning", "check": "constant",
                                 "message": f"'{col}' is constant."})

        log.info("Validation complete: %d issues, %d warnings",
                 len(issues), len(warnings))
        return {"issues": issues, "warnings": warnings, "passed": len(issues) == 0}