"""Class imbalance analysis and resampling strategies."""
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from config import RANDOM_STATE, TEST_SIZE
from logger import get_logger

log = get_logger(__name__)

class ImbalanceHandler:
    """Analyses imbalance and provides resampling strategies."""

    @staticmethod
    def analyse(y: pd.Series) -> dict:
        counts = y.value_counts()
        ratio = counts.max() / max(counts.min(), 1)
        return {
            "class_counts": counts.to_dict(),
            "imbalance_ratio": round(ratio, 2),
            "minority_pct": round(counts.min() / len(y) * 100, 4),
            "severity": ("Severe" if ratio > 50 else
                         "Moderate" if ratio > 10 else "Mild"),
        }

    @staticmethod
    def split_and_resample(X: pd.DataFrame, y: pd.Series,
                           strategy: str = "smote") -> Tuple:
        """Stratified split + optional resampling on training set only."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        log.info("Split: train=%d test=%d", len(X_train), len(X_test))

        if strategy == "smote":
            sampler = SMOTE(random_state=RANDOM_STATE)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        elif strategy == "smotetomek":
            sampler = SMOTETomek(random_state=RANDOM_STATE)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        elif strategy == "class_weight":
            # No resampling — return as-is; models use class_weight='balanced'
            X_res, y_res = X_train, y_train
        else:
            X_res, y_res = X_train, y_train

        log.info("Resampled train: %s → %s", y_train.value_counts().to_dict(),
                 y_res.value_counts().to_dict())
        return X_res, y_res, X_test, y_test

    @staticmethod
    def strategy_recommendation(ratio: float) -> str:
        if ratio < 5:
            return "Class weights (`class_weight='balanced'`) are sufficient."
        elif ratio < 10:
            return "Use class weights first; consider SMOTE if recall is low."
        elif ratio < 50:
            return "SMOTE or SMOTETomek with threshold tuning on the PR curve."
        else:
            return ("Severe imbalance — consider anomaly-detection framing "
                    "(Isolation Forest) in addition to supervised models.")