"""Behavioural feature engineering for fraud detection.

Every engineered feature is documented with its rationale.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from logger import get_logger

log = get_logger(__name__)

class FeatureEngineer:
    """Creates transaction-level behavioural features.

    Feature catalogue
    ─────────────────
    amount_deviation      : |amount - customer_mean| / customer_std
                            → How far this transaction deviates from the
                              customer's historical spending pattern.
    amount_percentile     : Percentile rank of the amount within the dataset.
                            → Captures unusually large or small transactions.
    time_since_prev       : Seconds since the previous transaction for the same
                            entity. Fraudsters often transact in rapid bursts.
    tx_count_daily        : Number of transactions by the entity on that day.
                            → High daily counts may indicate card testing.
    tx_velocity_hourly    : Transactions per hour for the entity.
                            → Velocity spikes are strong fraud signals.
    category_frequency    : Frequency of the transaction category.
                            → Rare categories may be higher risk.
    amount_to_balance     : Amount / (old_balance + 1) — proportion of balance
                            consumed. Fraudsters tend to drain accounts.
    balance_delta_orig    : new_balance_orig - old_balance_orig
                            → Inconsistent balance changes are suspicious.
    balance_delta_dest    : new_balance_dest - old_balance_dest
    is_round_amount       : 1 if amount is a round number, else 0.
                            → Fraudsters sometimes use round amounts.
    """

    @staticmethod
    def engineer(df: pd.DataFrame,
                 entity_col: Optional[str] = None,
                 time_col: Optional[str] = None,
                 amount_col: Optional[str] = None,
                 category_col: Optional[str] = None) -> pd.DataFrame:
        df = df.copy()
        created = []

        # ── Amount-based features ──────────────────────────────────
        if amount_col and amount_col in df.columns:
            df["amount_log"] = np.log1p(df[amount_col].clip(lower=0))
            df["amount_percentile"] = df[amount_col].rank(pct=True)
            df["is_round_amount"] = (
                df[amount_col] % 10 == 0).astype(int)
            created += ["amount_log", "amount_percentile", "is_round_amount"]

        # ── Entity-level aggregations ──────────────────────────────
        if entity_col and entity_col in df.columns and amount_col in df.columns:
            grp = df.groupby(entity_col)[amount_col]
            df["customer_amount_mean"] = grp.transform("mean")
            df["customer_amount_std"] = grp.transform("std").fillna(0)
            df["customer_amount_max"] = grp.transform("max")
            df["amount_deviation"] = np.where(
                df["customer_amount_std"] > 0,
                (df[amount_col] - df["customer_amount_mean"]) / df["customer_amount_std"],
                0.0,
            )
            created += ["customer_amount_mean", "customer_amount_std",
                        "customer_amount_max", "amount_deviation"]

        # ── Time-based features ────────────────────────────────────
        if time_col and time_col in df.columns:
            df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
            if entity_col and entity_col in df.columns:
                df = df.sort_values([entity_col, time_col])
                df["time_since_prev"] = df.groupby(entity_col)[time_col].diff().fillna(-1)
                created.append("time_since_prev")

            # Daily count
            df["day"] = (df[time_col] // 86400).astype(int)
            if entity_col and entity_col in df.columns:
                df["tx_count_daily"] = df.groupby([entity_col, "day"])[time_col].transform("count")
                created.append("tx_count_daily")

            # Hourly velocity
            df["hour_bucket"] = (df[time_col] // 3600).astype(int)
            if entity_col and entity_col in df.columns:
                df["tx_velocity_hourly"] = df.groupby([entity_col, "hour_bucket"])[time_col].transform("count")
                created.append("tx_velocity_hourly")

        # ── Category frequency ─────────────────────────────────────
        if category_col and category_col in df.columns:
            freq = df[category_col].value_counts(normalize=True)
            df["category_frequency"] = df[category_col].map(freq)
            created.append("category_frequency")

        # ── Balance-based features (PaySim-style) ──────────────────
        bal_pairs = [("oldbalanceOrg", "newbalanceOrg", "balance_delta_orig"),
                     ("oldbalanceDest", "newbalanceDest", "balance_delta_dest")]
        for old, new, name in bal_pairs:
            if old in df.columns and new in df.columns:
                df[name] = df[new] - df[old]
                created.append(name)

        if "oldbalanceOrg" in df.columns and amount_col in df.columns:
            df["amount_to_balance"] = df[amount_col] / (df["oldbalanceOrg"].abs() + 1)
            created.append("amount_to_balance")

        log.info("Engineered %d features: %s", len(created), created)
        return df

    @staticmethod
    def feature_catalogue() -> pd.DataFrame:
        """Return a human-readable catalogue of engineered features."""
        return pd.DataFrame([
            {"Feature": "amount_log", "Type": "Transformation",
             "Rationale": "Log-transform reduces skew in transaction amounts."},
            {"Feature": "amount_percentile", "Type": "Ranking",
             "Rationale": "Percentile rank captures relative size of transaction."},
            {"Feature": "is_round_amount", "Type": "Binary",
             "Rationale": "Round amounts are sometimes used by fraudsters."},
            {"Feature": "customer_amount_mean", "Type": "Aggregation",
             "Rationale": "Baseline spending level for the entity."},
            {"Feature": "customer_amount_std", "Type": "Aggregation",
             "Rationale": "Variability in entity spending."},
            {"Feature": "customer_amount_max", "Type": "Aggregation",
             "Rationale": "Largest historical transaction by the entity."},
            {"Feature": "amount_deviation", "Type": "Behavioural",
             "Rationale": "Z-score of amount vs. entity history — key fraud signal."},
            {"Feature": "time_since_prev", "Type": "Temporal",
             "Rationale": "Short gaps indicate rapid succession (card testing)."},
            {"Feature": "tx_count_daily", "Type": "Velocity",
             "Rationale": "Unusually high daily count may indicate fraud burst."},
            {"Feature": "tx_velocity_hourly", "Type": "Velocity",
             "Rationale": "Hourly velocity spikes are strong fraud indicators."},
            {"Feature": "category_frequency", "Type": "Frequency",
             "Rationale": "Rare categories may be higher risk."},
            {"Feature": "balance_delta_orig", "Type": "Balance",
             "Rationale": "Unexpected balance changes signal anomalies."},
            {"Feature": "balance_delta_dest", "Type": "Balance",
             "Rationale": "Receiver balance inconsistency is suspicious."},
            {"Feature": "amount_to_balance", "Type": "Ratio",
             "Rationale": "Proportion of balance consumed — fraud drains accounts."},
        ])