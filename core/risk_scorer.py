"""Combines model probability + anomaly flags into a unified risk score."""
import pandas as pd
import numpy as np
from config import RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD
from logger import get_logger

log = get_logger(__name__)

class RiskScorer:
    """Produces a 0–1 risk score and categorical risk band."""

    @staticmethod
    def score(fraud_prob: np.ndarray,
              anomaly_flag: np.ndarray,
              anomaly_score: np.ndarray) -> pd.DataFrame:
        # Normalise anomaly score to 0–1
        if anomaly_score.max() > anomaly_score.min():
            norm_anom = ((anomaly_score - anomaly_score.min()) /
                         (anomaly_score.max() - anomaly_score.min()))
        else:
            norm_anom = np.zeros_like(anomaly_score)

        risk = 0.75 * fraud_prob + 0.25 * norm_anom
        risk = np.clip(risk, 0, 1)

        bands = pd.cut(risk,
                       bins=[-0.01, RISK_MEDIUM_THRESHOLD,
                             RISK_HIGH_THRESHOLD, 1.01],
                       labels=["LOW", "MEDIUM", "HIGH"])

        return pd.DataFrame({
            "fraud_probability": fraud_prob,
            "anomaly_score": norm_anom,
            "anomaly_flag": anomaly_flag,
            "risk_score": risk,
            "risk_band": bands,
        })

    @staticmethod
    def explanation() -> str:
        return (
            "Risk score = 0.75 × supervised fraud probability + "
            "0.25 × normalised anomaly score. "
            "Bands: LOW < 0.40, MEDIUM 0.40–0.70, HIGH > 0.70. "
            "Thresholds are configurable in config.py."
        )