"""Unsupervised anomaly detection — Isolation Forest & statistical methods."""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
from config import ANOMALY_CONTAMINATION, RANDOM_STATE
from logger import get_logger

log = get_logger(__name__)

class AnomalyDetector:
    """Identifies 'potential anomalies' — never claims confirmed fraud."""

    def __init__(self, contamination: float = ANOMALY_CONTAMINATION):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_estimators=200,
            n_jobs=-1,
        )

    def fit_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with anomaly flags and scores."""
        X_num = X.select_dtypes(include=[np.number]).fillna(0)
        self.model.fit(X_num)
        flags = self.model.predict(X_num)           # -1 = anomaly, 1 = normal
        scores = -self.model.decision_function(X_num)  # higher = more anomalous

        result = pd.DataFrame({
            "isolation_forest_flag": (flags == -1).astype(int),
            "isolation_forest_score": scores,
        }, index=X.index)

        # Statistical: z-score on amount (if present)
        if "Amount" in X.columns:
            z = np.abs(stats.zscore(X["Amount"].fillna(0)))
            result["statistical_zscore"] = z
            result["statistical_flag"] = (z > 3).astype(int)
        elif "amount" in X.columns:
            z = np.abs(stats.zscore(X["amount"].fillna(0)))
            result["statistical_zscore"] = z
            result["statistical_flag"] = (z > 3).astype(int)

        result["combined_anomaly_flag"] = (
            (result["isolation_forest_flag"] == 1) |
            (result.get("statistical_flag", 0) == 1)
        ).astype(int)

        log.info("Anomaly detection complete: %d anomalies",
                 result["combined_anomaly_flag"].sum())
        return result

    @staticmethod
    def interpretation() -> str:
        return (
            "IMPORTANT: Anomaly detection identifies transactions that are "
            "statistically unusual. 'Potential anomaly' ≠ 'Confirmed fraud'. "
            "These flags should be used to prioritise human investigation, "
            "not as evidence of criminal activity."
        )