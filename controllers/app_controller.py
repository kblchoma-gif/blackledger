"""Orchestrates the full pipeline from import to investigation.

BLACKLEDGER — AppController

Fixes applied:
  • feature_columns_ is captured at training time and reused everywhere
    downstream (anomaly detection, investigation) to keep the feature
    matrix identical between fit and predict.
  • load_investigation() selects a model by name instead of relying on
    Python dict insertion order.
  • save_metrics is wrapped so a stale SQLite schema cannot kill a
    successful training run.
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import COLORS, FONTS
from logger import get_logger

from core.data_loader import DataLoader
from core.data_validator import DataValidator
from core.data_cleaner import DataCleaner
from core.feature_engineer import FeatureEngineer
from core.imbalance_handler import ImbalanceHandler
from core.model_trainer import ModelTrainer
from core.model_evaluator import ModelEvaluator
from core.anomaly_detector import AnomalyDetector
from core.risk_scorer import RiskScorer
from core.exporter import Exporter
from database.db_manager import DatabaseManager
from utils import plot_utils

log = get_logger(__name__)


class AppController:
    """Central controller wiring the UI to the analysis pipeline."""

    def __init__(self, view):
        self.view = view
        self.db = DatabaseManager()

        # Data pipeline state
        self.df_raw = None
        self.df_clean = None
        self.df_feat = None
        self.target_col = None

        # Model / evaluation state
        self.models = {}
        self.metrics_df = None
        self.feature_columns_ = None      # exact column order used to fit

        # Post-model state
        self.anomaly_df = None
        self.risk_df = None
        self.final_df = None

    # ══════════════════════════════════════════════════════════════
    #  DATA IMPORT
    # ══════════════════════════════════════════════════════════════

    def import_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.df_raw = DataLoader.load_csv(Path(path))
            self._after_import(path)
        except Exception as exc:
            log.exception("Import failed")
            messagebox.showerror("Import Error", str(exc))

    def load_sample(self):
        """Generate a synthetic sample resembling the PaySim dataset."""
        rng = np.random.default_rng(42)
        n = 5000
        types = rng.choice(["CASH_IN", "CASH_OUT", "PAYMENT",
                            "TRANSFER", "DEBIT"], n,
                           p=[0.15, 0.30, 0.35, 0.15, 0.05])
        amounts = rng.lognormal(7, 1.5, n).round(2)
        old_bal_o = rng.lognormal(10, 1, n).round(2)
        new_bal_o = np.maximum(old_bal_o - amounts, 0).round(2)
        old_bal_d = rng.lognormal(10, 1, n).round(2)
        new_bal_d = (old_bal_d + amounts).round(2)
        is_fraud = (rng.random(n) < 0.012).astype(int)

        df = pd.DataFrame({
            "step": rng.integers(1, 744, n),
            "type": types,
            "amount": amounts,
            "nameOrig": [f"C{i:08d}" for i in rng.integers(0, n // 2, n)],
            "oldbalanceOrg": old_bal_o,
            "newbalanceOrg": new_bal_o,
            "nameDest": [f"M{i:08d}" for i in rng.integers(0, n // 3, n)],
            "oldbalanceDest": old_bal_d,
            "newbalanceDest": new_bal_d,
            "isFraud": is_fraud,
        })
        self.df_raw = df
        self._after_import("synthetic_sample")

    def _after_import(self, source):
        self.target_col = DataLoader.detect_target_column(self.df_raw)
        if self.target_col is None:
            messagebox.showwarning(
                "Target Not Found",
                "Could not identify a binary fraud label in the dataset.")
            return

        profile = DataLoader.profile(self.df_raw)
        self._display_profile(source, profile)
        self._run_validation()
        self._clean()
        self._engineer()
        self._display_eda()

        # Reset downstream state so the user can't investigate stale results.
        self.models = {}
        self.metrics_df = None
        self.feature_columns_ = None
        self.anomaly_df = None
        self.risk_df = None
        self.final_df = None

    # ══════════════════════════════════════════════════════════════
    #  VALIDATION & CLEANING
    # ══════════════════════════════════════════════════════════════

    def _run_validation(self):
        result = DataValidator.validate(self.df_raw, self.target_col)
        text = self.view.quality_text
        text.delete("1.0", "end")
        text.insert("end", "DATA QUALITY REPORT\n" + "=" * 60 + "\n\n")
        text.insert("end", f"Target column: {self.target_col}\n")
        text.insert("end", f"Passed: {result['passed']}\n\n")
        if result["issues"]:
            text.insert("end", "ERRORS:\n")
            for i in result["issues"]:
                text.insert("end",
                            f"  [{i['level'].upper()}] "
                            f"{i['check']}: {i['message']}\n")
        if result["warnings"]:
            text.insert("end", "\nWARNINGS:\n")
            for w in result["warnings"]:
                text.insert("end",
                            f"  [{w['level'].upper()}] "
                            f"{w['check']}: {w['message']}\n")

    def _clean(self):
        self.df_clean = DataCleaner.clean(self.df_raw, self.target_col)
        log.info("Cleaned shape: %s", self.df_clean.shape)

    def _engineer(self):
        df = self.df_clean

        entity = None
        for c in ["nameOrig", "customer_id", "user_id", "account_id"]:
            if c in df.columns:
                entity = c
                break

        time_c = "step" if "step" in df.columns else (
            "Time" if "Time" in df.columns else None)
        amount_c = "amount" if "amount" in df.columns else (
            "Amount" if "Amount" in df.columns else None)
        cat_c = "type" if "type" in df.columns else None

        self.df_feat = FeatureEngineer.engineer(
            df,
            entity_col=entity,
            time_col=time_c,
            amount_col=amount_c,
            category_col=cat_c)

        catalogue = FeatureEngineer.feature_catalogue()
        text = self.view.features_text
        text.delete("1.0", "end")
        text.insert("end", "ENGINEERED FEATURE CATALOGUE\n"
                          + "=" * 60 + "\n\n")
        for _, row in catalogue.iterrows():
            text.insert("end", f"• {row['Feature']}  [{row['Type']}]\n")
            text.insert("end", f"  {row['Rationale']}\n\n")

    # ══════════════════════════════════════════════════════════════
    #  DISPLAY HELPERS
    # ══════════════════════════════════════════════════════════════

    def _display_profile(self, source, profile):
        text = self.view.dataset_info
        text.delete("1.0", "end")
        text.insert("end", f"SOURCE: {source}\n" + "=" * 60 + "\n\n")
        text.insert("end", f"Rows: {profile['n_rows']:,}\n")
        text.insert("end", f"Columns: {profile['n_cols']}\n")
        text.insert("end", f"Memory: {profile['memory_mb']} MB\n")
        text.insert("end", f"Duplicates: {profile['n_duplicates']}\n\n")
        text.insert("end", f"Target column: {self.target_col}\n\n")
        text.insert("end", "COLUMNS:\n")
        for col, dt in profile["dtypes"].items():
            miss = profile["missing_pct"].get(col, 0)
            text.insert("end", f"  {col:<20} {dt:<12} missing={miss}%\n")

    def _display_eda(self):
        # Clear previous widgets in the EDA tab
        for w in self.view.eda_frame.winfo_children():
            w.destroy()

        df = self.df_feat
        target = self.target_col

        # ── Text summary ──────────────────────────────────────────
        summary = tk.Text(self.view.eda_frame, bg=COLORS["bg_mid"],
                          fg=COLORS["fg_primary"], font=FONTS["mono"],
                          relief="flat", height=10)
        summary.pack(fill="x", padx=4, pady=4)
        summary.insert("end", "EXPLORATORY SUMMARY\n" + "=" * 50 + "\n")
        summary.insert("end", f"Shape: {df.shape}\n")
        summary.insert("end",
                       f"Target distribution:\n"
                       f"{df[target].value_counts().to_string()}\n\n")
        num = df.select_dtypes(include=[np.number])
        summary.insert("end", "Numeric describe (top 10):\n")
        summary.insert("end", num.describe().T.head(10).to_string())
        summary.config(state="disabled")

        # ── Fraud distribution chart ──────────────────────────────
        fig1 = plot_utils.plot_fraud_distribution(df, target)
        self._embed_fig(fig1, self.view.eda_frame)

    def _embed_fig(self, fig, parent):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True,
                                    padx=4, pady=4)

    # ══════════════════════════════════════════════════════════════
    #  TRAINING
    # ══════════════════════════════════════════════════════════════

    def train_models(self):
        if self.df_feat is None:
            messagebox.showwarning("No Data", "Import a dataset first.")
            return

        target = self.target_col

        # ── Build the exact feature matrix models will be fit on ──
        X = (self.df_feat
             .drop(columns=[target], errors="ignore")
             .select_dtypes(include=[np.number]))
        y = self.df_feat[target]

        if X.empty:
            messagebox.showerror(
                "No Features",
                "No numeric features available after engineering.")
            return

        # Remember the column order — reused downstream in
        # run_anomaly() and load_investigation().
        self.feature_columns_ = X.columns.tolist()
        log.info("Training on %d features: %s",
                 len(self.feature_columns_), self.feature_columns_)

        strategy = self.view.imbalance_var.get()
        X_res, y_res, X_test, y_test = ImbalanceHandler.split_and_resample(
            X, y, strategy=strategy)

        trainer = ModelTrainer()
        self.models = trainer.train_all(
            X_res, y_res,
            use_class_weight=(strategy != "smote"))

        self.metrics_df = ModelEvaluator.evaluate_all(
            self.models, X_test, y_test)

        # Persist metrics — do not let a stale schema kill training.
        try:
            self.db.save_metrics(self.metrics_df)
        except Exception as exc:
            log.warning("Could not persist metrics to SQLite: %s", exc)

        # ── Report to Training tab ────────────────────────────────
        text = self.view.training_text
        text.delete("1.0", "end")
        text.insert("end", "TRAINING COMPLETE\n" + "=" * 60 + "\n\n")
        text.insert("end", f"Imbalance strategy: {strategy}\n")
        text.insert("end", f"Training samples: {len(X_res):,}\n")
        text.insert("end", f"Test samples: {len(X_test):,}\n")
        text.insert("end", f"Features used: {len(self.feature_columns_)}\n\n")
        text.insert("end", self.metrics_df.to_string(index=False))

        self._display_evaluation()

    def _display_evaluation(self):
        for w in self.view.eval_frame.winfo_children():
            w.destroy()

        if self.metrics_df is None:
            return

        # ── Metrics table ─────────────────────────────────────────
        tbl = tk.Text(self.view.eval_frame, bg=COLORS["bg_mid"],
                      fg=COLORS["fg_primary"], font=FONTS["mono"],
                      relief="flat", height=8)
        tbl.pack(fill="x", padx=4, pady=4)
        tbl.insert("end", "MODEL COMPARISON\n" + "=" * 70 + "\n")
        tbl.insert("end", self.metrics_df.to_string(index=False))
        tbl.insert("end",
                   "\n\nNOTE: Accuracy is deliberately omitted. "
                   "For imbalanced fraud data, PR-AUC and F1 are more "
                   "informative than ROC-AUC or accuracy.")
        tbl.config(state="disabled")

        # ── Model comparison chart ────────────────────────────────
        fig = plot_utils.plot_model_comparison(self.metrics_df)
        self._embed_fig(fig, self.view.eval_frame)

    # ══════════════════════════════════════════════════════════════
    #  ANOMALY DETECTION
    # ══════════════════════════════════════════════════════════════

    def run_anomaly(self):
        if self.df_feat is None:
            messagebox.showwarning("No Data", "Import a dataset first.")
            return

        # Use the same feature matrix models were trained on. If
        # training has not run yet, fall back to all numeric columns
        # minus the target.
        if self.feature_columns_ is not None:
            X = self.df_feat[self.feature_columns_]
        else:
            X = (self.df_feat
                 .drop(columns=[self.target_col], errors="ignore")
                 .select_dtypes(include=[np.number]))

        detector = AnomalyDetector()
        self.anomaly_df = detector.fit_predict(X)

        text = self.view.anomaly_text
        text.delete("1.0", "end")
        text.insert("end", "ANOMALY DETECTION RESULTS\n"
                          + "=" * 60 + "\n\n")
        text.insert("end", AnomalyDetector.interpretation() + "\n\n")
        text.insert("end", f"Total transactions: {len(self.anomaly_df):,}\n")
        text.insert("end", f"Isolation Forest anomalies: "
                          f"{self.anomaly_df['isolation_forest_flag'].sum():,}\n")
        if "statistical_flag" in self.anomaly_df.columns:
            text.insert("end", f"Statistical (z>3) anomalies: "
                              f"{self.anomaly_df['statistical_flag'].sum():,}\n")
        text.insert("end", f"Combined anomalies: "
                          f"{self.anomaly_df['combined_anomaly_flag'].sum():,}\n")

    # ══════════════════════════════════════════════════════════════
    #  INVESTIGATION
    # ══════════════════════════════════════════════════════════════

    def load_investigation(self):
        if self.anomaly_df is None:
            messagebox.showwarning(
                "Anomaly",
                "Run anomaly detection before loading investigations.")
            return

        if self.feature_columns_ is None or not self.models:
            messagebox.showwarning(
                "Train First",
                "Train the models before loading investigations.")
            return

        # ── Use the exact same feature matrix as training ─────────
        X = self.df_feat[self.feature_columns_]

        # ── Select a model by preference, not dict order ──────────
        preference = ("Gradient Boosting", "Random Forest",
                      "Decision Tree", "Logistic Regression")
        model_name = next(
            (n for n in preference if n in self.models),
            next(iter(self.models)),
        )
        model = self.models[model_name]
        log.info("Scoring investigations with %s", model_name)

        prob = model.predict_proba(X)[:, 1]

        self.risk_df = RiskScorer.score(
            prob,
            self.anomaly_df["combined_anomaly_flag"].values,
            self.anomaly_df["isolation_forest_score"].values)

        self.final_df = pd.concat(
            [self.df_feat.reset_index(drop=True),
             self.risk_df.reset_index(drop=True)],
            axis=1)

        # ── Populate tree with HIGH-band rows ─────────────────────
        high = self.final_df[self.final_df["risk_band"] == "HIGH"]
        tree = self.view.inv_tree
        for item in tree.get_children():
            tree.delete(item)

        for idx, row in high.head(200).iterrows():
            tree.insert("", "end", iid=str(idx),
                        values=(idx,
                                f"{row['risk_score']:.4f}",
                                row["risk_band"],
                                f"{row['fraud_probability']:.4f}",
                                "YES" if row["anomaly_flag"] else "NO"))

        log.info("Investigation loaded: %d high-risk transactions", len(high))

    def on_transaction_select(self, event):
        sel = self.view.inv_tree.selection()
        if not sel or self.final_df is None:
            return

        idx = int(sel[0])
        if idx not in self.final_df.index:
            return

        row = self.final_df.loc[idx]
        text = self.view.inv_detail
        text.delete("1.0", "end")
        text.insert("end", f"TRANSACTION #{idx}\n" + "=" * 50 + "\n\n")

        for col in row.index:
            val = row[col]
            if isinstance(val, float):
                val = f"{val:.6f}"
            text.insert("end", f"{col:<25} {val}\n")

        text.insert("end", "\n" + RiskScorer.explanation())

    # ══════════════════════════════════════════════════════════════
    #  VISUALISATIONS
    # ══════════════════════════════════════════════════════════════

    def generate_charts(self):
        for w in self.view.viz_frame.winfo_children():
            w.destroy()

        if self.df_feat is None:
            return

        df = self.df_feat
        target = self.target_col
        amount_c = "amount" if "amount" in df.columns else (
            "Amount" if "Amount" in df.columns else None)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        charts = []
        charts.append(plot_utils.plot_fraud_distribution(df, target))

        if amount_c:
            charts.append(plot_utils.plot_amount_distribution(
                df, amount_c, target))

        if len(num_cols) > 1:
            charts.append(plot_utils.plot_correlation_heatmap(
                df, num_cols[:15]))

        if self.anomaly_df is not None:
            charts.append(plot_utils.plot_anomaly_distribution(
                self.anomaly_df))

        for fig in charts:
            self._embed_fig(fig, self.view.viz_frame)

    # ══════════════════════════════════════════════════════════════
    #  EXPORT
    # ══════════════════════════════════════════════════════════════

    def export(self, kind: str):
        log.info("Export requested: %s", kind)
        log_widget = self.view.export_log
        log_widget.insert("end", f"[{kind}] Export requested.\n")

        try:
            if kind == "clean_csv" and self.df_clean is not None:
                p = Exporter.to_csv(self.df_clean, "clean_dataset")
                log_widget.insert("end", f"  → {p}\n")

            elif kind == "pred_csv" and self.final_df is not None:
                p = Exporter.to_csv(self.final_df, "predictions")
                log_widget.insert("end", f"  → {p}\n")

            elif kind == "risk_csv" and self.risk_df is not None:
                p = Exporter.to_csv(self.risk_df, "risk_scores")
                log_widget.insert("end", f"  → {p}\n")

            elif kind == "anomaly_csv" and self.anomaly_df is not None:
                p = Exporter.to_csv(self.anomaly_df, "anomalies")
                log_widget.insert("end", f"  → {p}\n")

            elif kind == "metrics_json" and self.metrics_df is not None:
                p = Exporter.to_json(self.metrics_df, "model_metrics")
                log_widget.insert("end", f"  → {p}\n")

            elif kind == "summary" and self.df_feat is not None:
                p = Exporter.summary_report(self.df_feat, "summary")
                log_widget.insert("end", f"  → {p}\n")

            elif kind == "charts_png":
                log_widget.insert(
                    "end",
                    "  (Charts are exported individually from the "
                    "Visualisations tab.)\n")

            else:
                log_widget.insert(
                    "end",
                    "  ⚠  Data not ready for this export.\n")

        except Exception as exc:
            log.exception("Export failed")
            log_widget.insert("end", f"  ✗ Error: {exc}\n")