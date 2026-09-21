"""Comprehensive evaluation with multiple metrics."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)
from logger import get_logger

log = get_logger(__name__)

class ModelEvaluator:
    """Computes and stores evaluation metrics for all models."""

    @staticmethod
    def evaluate(model, X_test: pd.DataFrame,
                 y_test: pd.Series, name: str) -> Dict[str, Any]:
        y_pred = model.predict(X_test)
        y_prob = (model.predict_proba(X_test)[:, 1]
                  if hasattr(model, "predict_proba") else y_pred)

        metrics = {
            "model": name,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": average_precision_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "y_prob": y_prob,
        }
        log.info("%s — F1=%.4f ROC-AUC=%.4f PR-AUC=%.4f",
                 name, metrics["f1"], metrics["roc_auc"], metrics["pr_auc"])
        return metrics

    @staticmethod
    def evaluate_all(models: Dict[str, Any], X_test: pd.DataFrame,
                     y_test: pd.Series) -> pd.DataFrame:
        rows = []
        for name, model in models.items():
            res = ModelEvaluator.evaluate(model, X_test, y_test, name)
            rows.append({k: v for k, v in res.items()
                         if k not in ("confusion_matrix", "y_prob")})
        return pd.DataFrame(rows)

    @staticmethod
    def full_report(model, X_test: pd.DataFrame,
                    y_test: pd.Series, name: str) -> str:
        y_pred = model.predict(X_test)
        return classification_report(y_test, y_pred, zero_division=0)

    @staticmethod
    def curve_data(model, X_test: pd.DataFrame, y_test: pd.Series):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        return {"roc": (fpr, tpr), "pr": (rec, prec)}