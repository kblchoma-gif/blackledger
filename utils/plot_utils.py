"""Dark-themed Matplotlib/Seaborn plots for BLACKLEDGER."""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from config import COLORS

def apply_dark_theme():
    """Apply Wednesday Addams dark theme to all plots."""
    sns.set_theme(style="darkgrid", palette="dark")
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg_dark"],
        "axes.facecolor": COLORS["bg_mid"],
        "axes.edgecolor": COLORS["border"],
        "axes.labelcolor": COLORS["fg_primary"],
        "text.color": COLORS["fg_primary"],
        "xtick.color": COLORS["fg_secondary"],
        "ytick.color": COLORS["fg_secondary"],
        "grid.color": COLORS["border"],
        "grid.alpha": 0.3,
        "font.family": "monospace",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.dpi": 100,
    })

def fig_to_canvas(fig):
    """Convert a Matplotlib figure to a Tkinter-compatible canvas."""
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    return FigureCanvasTkAgg(fig, master=None)

# ── Individual plots ───────────────────────────────────────────────

def plot_fraud_distribution(df, target_col):
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[target_col].value_counts()
    labels = ["Legitimate", "Fraud"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    bars = ax.bar(labels, values,
                  color=[COLORS["info"], COLORS["accent"]],
                  edgecolor=COLORS["border"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{v:,}", ha="center", va="bottom",
                color=COLORS["fg_primary"], fontsize=10, fontweight="bold")
    ax.set_title("Fraud Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig

def plot_amount_distribution(df, amount_col, target_col):
    apply_dark_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cls, label, colour in zip(
            axes, [0, 1], ["Legitimate", "Fraud"],
            [COLORS["info"], COLORS["accent"]]):
        subset = df[df[target_col] == cls][amount_col].dropna()
        ax.hist(subset, bins=50, color=colour, alpha=0.8,
                edgecolor=COLORS["border"])
        ax.set_title(f"{label} — Amount")
        ax.set_xlabel("Amount")
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig

def plot_correlation_heatmap(df, numeric_cols):
    apply_dark_theme()
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.7}, ax=ax,
                annot=False)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig

def plot_confusion_matrix(cm, labels=None):
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=labels or ["Pred 0", "Pred 1"],
                yticklabels=labels or ["True 0", "True 1"],
                ax=ax, cbar=False)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig

def plot_roc_curve(fpr, tpr, auc_val):
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=COLORS["accent"], lw=2,
            label=f"ROC (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=COLORS["fg_muted"], lw=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig

def plot_pr_curve(recall, precision, auc_val):
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=COLORS["accent"], lw=2,
            label=f"PR (AUC = {auc_val:.3f})")
    ax.set_title("Precision–Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, top_n=15):
    apply_dark_theme()
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
        if hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_[0])
        else:
            imp = model.named_steps["clf"].feature_importances_
    else:
        return None
    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in idx], imp[idx],
            color=COLORS["accent"], edgecolor=COLORS["border"])
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig

def plot_model_comparison(metrics_df):
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    x = np.arange(len(metrics_df))
    width = 0.15
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, metrics_df[m], width, label=m.upper())
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metrics_df["model"], rotation=15)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig

def plot_anomaly_distribution(anomaly_df):
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = anomaly_df["combined_anomaly_flag"].value_counts()
    bars = ax.bar(["Normal", "Anomaly"],
                  [counts.get(0, 0), counts.get(1, 0)],
                  color=[COLORS["info"], COLORS["accent"]],
                  edgecolor=COLORS["border"])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{int(bar.get_height()):,}", ha="center",
                va="bottom", color=COLORS["fg_primary"],
                fontweight="bold")
    ax.set_title("Anomaly Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig