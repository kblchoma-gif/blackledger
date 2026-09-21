"""Multi-model training with reproducibility and persistence."""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config import MODEL_DIR, RANDOM_STATE
from logger import get_logger

log = get_logger(__name__)

class ModelTrainer:
    """Trains and persists four classifiers."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()

    def _build_models(self, use_class_weight: bool = True) -> Dict[str, Any]:
        cw = "balanced" if use_class_weight else None
        return {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight=cw, max_iter=2000,
                    random_state=RANDOM_STATE, n_jobs=-1)),
            ]),
            "Decision Tree": DecisionTreeClassifier(
                class_weight=cw, max_depth=8,
                random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, class_weight=cw,
                max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1,
                max_depth=5, random_state=RANDOM_STATE),
        }

    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series,
                  use_class_weight: bool = True) -> Dict[str, Any]:
        models = self._build_models(use_class_weight)
        for name, model in models.items():
            log.info("Training %s …", name)
            model.fit(X_train, y_train)
            self.models[name] = model
            joblib.dump(model, MODEL_DIR / f"{name.replace(' ', '_').lower()}.joblib",
                        compress=3)
            log.info("Saved %s", name)
        return self.models

    def load_model(self, name: str):
        path = MODEL_DIR / f"{name.replace(' ', '_').lower()}.joblib"
        model = joblib.load(path)
        log.info("Loaded model from %s", path)
        return model