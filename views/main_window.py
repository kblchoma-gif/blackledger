"""BLACKLEDGER main window with Notebook tabs.

Fix: MainWindow.__init__ now accepts controller=None and every button
command uses a deferred lambda, so the controller can be wired in after
the view is built (standard MVC "wire after construction" pattern).
"""
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

from config import COLORS, FONTS
from logger import get_logger

log = get_logger(__name__)


class MainWindow:
    """Root application window with dark Wednesday Addams theme."""

    def __init__(self, controller=None):
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("BLACKLEDGER — Financial Fraud Detection & Investigation Engine")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 750)
        self.root.configure(bg=COLORS["bg_darkest"])
        self._setup_styles()
        self._build_header()
        self._build_notebook()

    # ── Styles ─────────────────────────────────────────────────────

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",
                        background=COLORS["bg_dark"],
                        borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=COLORS["bg_mid"],
                        foreground=COLORS["fg_secondary"],
                        padding=[18, 8],
                        font=FONTS["body"])
        style.map("TNotebook.Tab",
                  background=[("selected", COLORS["bg_light"])],
                  foreground=[("selected", COLORS["accent"])])
        style.configure("TFrame", background=COLORS["bg_dark"])
        style.configure("TLabel",
                        background=COLORS["bg_dark"],
                        foreground=COLORS["fg_primary"],
                        font=FONTS["body"])
        style.configure("Heading.TLabel",
                        background=COLORS["bg_dark"],
                        foreground=COLORS["accent"],
                        font=FONTS["heading"])
        style.configure("TButton",
                        background=COLORS["bg_light"],
                        foreground=COLORS["fg_primary"],
                        font=FONTS["body"],
                        padding=[12, 6])
        style.map("TButton",
                  background=[("active", COLORS["accent_soft"])])
        style.configure("Treeview",
                        background=COLORS["bg_mid"],
                        foreground=COLORS["fg_primary"],
                        fieldbackground=COLORS["bg_mid"],
                        font=FONTS["mono"],
                        rowheight=22)
        style.configure("Treeview.Heading",
                        background=COLORS["bg_light"],
                        foreground=COLORS["accent"],
                        font=FONTS["subheading"])
        style.map("Treeview", background=[("selected", COLORS["accent_soft"])])

    # ── Header ─────────────────────────────────────────────────────

    def _build_header(self):
        header = tk.Frame(self.root, bg=COLORS["bg_darkest"], height=70)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        title = tk.Label(header, text="BLACKLEDGER",
                         font=FONTS["title"],
                         fg=COLORS["accent"],
                         bg=COLORS["bg_darkest"])
        title.pack(side="left", padx=24, pady=12)

        subtitle = tk.Label(header,
                            text="Financial Fraud Detection & Investigation Engine",
                            font=("Courier New", 10, "italic"),
                            fg=COLORS["fg_muted"],
                            bg=COLORS["bg_darkest"])
        subtitle.pack(side="left", pady=(24, 0))

        quote = tk.Label(header,
                         text='"If the transaction looks innocent, examine it twice."',
                         font=("Courier New", 9, "italic"),
                         fg=COLORS["fg_muted"],
                         bg=COLORS["bg_darkest"])
        quote.pack(side="right", padx=24)

        sep = tk.Frame(self.root, bg=COLORS["border"], height=1)
        sep.pack(fill="x")

    # ── Notebook ───────────────────────────────────────────────────

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=8)

        self.tabs = {}
        tab_names = [
            ("dataset", "Dataset"),
            ("quality", "Data Quality"),
            ("eda", "Exploratory Analysis"),
            ("features", "Feature Engineering"),
            ("training", "Model Training"),
            ("evaluation", "Model Evaluation"),
            ("anomaly", "Anomaly Detection"),
            ("investigation", "Investigation"),
            ("visualisations", "Visualisations"),
            ("export", "Export"),
        ]
        for key, label in tab_names:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=f"  {label}  ")
            self.tabs[key] = frame

        self._build_dataset_tab()
        self._build_quality_tab()
        self._build_eda_tab()
        self._build_features_tab()
        self._build_training_tab()
        self._build_evaluation_tab()
        self._build_anomaly_tab()
        self._build_investigation_tab()
        self._build_visualisations_tab()
        self._build_export_tab()

    # ── Tab: Dataset ───────────────────────────────────────────────

    def _build_dataset_tab(self):
        f = self.tabs["dataset"]
        ttk.Label(f, text="DATASET IMPORT & PROFILING",
                  style="Heading.TLabel").pack(anchor="w", padx=16, pady=(16, 8))

        btn_frame = ttk.Frame(f)
        btn_frame.pack(fill="x", padx=16, pady=4)
        ttk.Button(btn_frame, text="Import CSV…",
                   command=lambda: self.controller.import_csv()
                   ).pack(side="left", padx=(0, 8))
        ttk.Button(btn_frame, text="Load Sample Dataset",
                   command=lambda: self.controller.load_sample()
                   ).pack(side="left")

        self.dataset_info = tk.Text(f, bg=COLORS["bg_mid"],
                                    fg=COLORS["fg_primary"],
                                    font=FONTS["mono"],
                                    insertbackground=COLORS["fg_primary"],
                                    relief="flat", wrap="word")
        self.dataset_info.pack(fill="both", expand=True, padx=16, pady=8)

    # ── Tab: Data Quality ──────────────────────────────────────────

    def _build_quality_tab(self):
        f = self.tabs["quality"]
        ttk.Label(f, text="DATA QUALITY REPORT",
                  style="Heading.TLabel").pack(anchor="w", padx=16, pady=(16, 8))
        self.quality_text = tk.Text(f, bg=COLORS["bg_mid"],
                                    fg=COLORS["fg_primary"],
                                    font=FONTS["mono"],
                                    relief="flat", wrap="word")
        self.quality_text.pack(fill="both", expand=True, padx=16, pady=8)

    # ── Tab: EDA ───────────────────────────────────────────────────

    def _build_eda_tab(self):
        f = self.tabs["eda"]
        ttk.Label(f, text="EXPLORATORY DATA ANALYSIS",
                  style="Heading.TLabel").pack(anchor="w", padx=16, pady=(16, 8))
        self.eda_frame = ttk.Frame(f)
        self.eda_frame.pack(fill="both", expand=True, padx=16, pady=8)

    # ── Tab: Feature Engineering ───────────────────────────────────

    def _build_features_tab(self):
        f = self.tabs["features"]
        ttk.Label(f, text="FEATURE ENGINEERING CATALOGUE",
                  style="Heading.TLabel").pack(anchor="w", padx=16, pady=(16, 8))
        self.features_text = tk.Text(f, bg=COLORS["bg_mid"],
                                     fg=COLORS["fg_primary"],
                                     font=FONTS["mono"],
                                     relief="flat", wrap="word")
        self.features_text.pack(fill="both", expand=True, padx=16, pady=8)

    # ── Tab: Training ──────────────────────────────────────────────

    def _build_training_tab(self):
        f = self.tabs["training"]
        ttk.Label(f, text="MODEL TRAINING",
                  style="Heading.TLabel").pack(anchor="w", padx=16, pady=(16, 8))

        ctrl = ttk.Frame(f)
        ctrl.pack(fill="x", padx=16, pady=4)
        ttk.Label(ctrl, text="Imbalance strategy:").pack(side="left")
        self.imbalance_var = tk.StringVar(value="smote")
        ttk.Combobox(ctrl, textvariable=self.imbalance_var,
                     values=["smote", "smotetomek", "class_weight", "none"],
                     state="readonly", width=14).pack(side="left", padx=8)
        ttk.Button(ctrl, text="Train All Models",
                   command=lambda: self.controller.train_models()
                   ).pack(side="left", padx=8)

        self.training_text = tk.Text(f, bg=COLORS["bg_mid"],
                                     fg=COLORS["fg_primary"],
                                     font=FONTS["mono"],
                                     relief="flat", wrap="word")
        self.training_text.pack(fill="both", expand=True, padx=16, pady=8)

    # ── Tab: Evaluation ────────────────────────────────────────────

    def _build_evaluation_tab(self):
        f = self.tabs["evaluation"]
        ttk.Label(f, text="MODEL EVALUATION",
                  style="Heading.TLabel").pack(anchor="w", padx=16, pady=(16, 8))
        self.eval_frame = ttk.Frame(f)
        self.eval_frame.pack(fill="both", expand=True, padx=16, pady=8)

    # ── Tab: Anomaly Detection ─────────────────────────────────────

    def _build_anomaly_tab(self):
        f = self.tabs["anomaly"]
        ttk.Label(f, text="ANOMALY DETECTION",
                  style="Heading.TLabel").pack(anchor="w", padx=16, pady=(16, 8))
        note = tk.Label(f,
                        text=("⚠  'Potential anomaly' ≠ 'Confirmed fraud'. "
                              "Flags prioritise human investigation."),
                        font=("Courier New", 9, "italic"),
                        fg=COLORS["warning"], bg=COLORS["bg_dark"])
        note.pack(anchor="w", padx=16)
        ttk.Button(f, text="Run Anomaly Detection",
                   command=lambda: self.controller.run_anomaly()
                   ).pack(anchor="w", padx=16, pady=8)
        self.anomaly_text = tk.Text(f, bg=COLORS["bg_mid"],
                                    fg=COLORS["fg_primary"],
                                    font=FONTS["mono"],
                                    relief="flat", wrap="word")
        self.anomaly_text.pack(fill="both", expand=True, padx=16, pady=8)

    # ── Tab: Investigation ─────────────────────────────────────────

    def _build_investigation_tab(self):
        f = self.tabs["investigation"]
        ttk.Label(f, text="INVESTIGATION — SUSPICIOUS TRANSACTIONS",
                  style="Heading.TLabel").pack(anchor="w", padx=16, pady=(16, 8))
        ttk.Button(f, text="Load High-Risk Transactions",
                   command=lambda: self.controller.load_investigation()
                   ).pack(anchor="w", padx=16, pady=4)

        cols = ("idx", "risk", "band", "prob", "anomaly")
        self.inv_tree = ttk.Treeview(f, columns=cols,
                                     show="headings", height=14)
        for c, h in zip(cols,
                        ["Index", "Risk Score", "Band",
                         "Fraud Prob", "Anomaly"]):
            self.inv_tree.heading(c, text=h)
            self.inv_tree.column(c, width=130, anchor="center")
        self.inv_tree.pack(fill="both", expand=True, padx=16, pady=4)
        self.inv_tree.bind("<<TreeviewSelect>>",
                           lambda e: self.controller.on_transaction_select(e))

        self.inv_detail = tk.Text(f, bg=COLORS["bg_mid"],
                                  fg=COLORS["fg_primary"],
                                  font=FONTS["mono"],
                                  relief="flat", wrap="word", height=10)
        self.inv_detail.pack(fill="both", expand=True, padx=16, pady=8)

    # ── Tab: Visualisations ────────────────────────────────────────

    def _build_visualisations_tab(self):
        f = self.tabs["visualisations"]
        ttk.Label(f, text="VISUALISATIONS",
                  style="Heading.TLabel").pack(anchor="w", padx=16, pady=(16, 8))
        btn = ttk.Frame(f)
        btn.pack(fill="x", padx=16, pady=4)
        ttk.Button(btn, text="Generate All Charts",
                   command=lambda: self.controller.generate_charts()
                   ).pack(side="left")
        self.viz_frame = ttk.Frame(f)
        self.viz_frame.pack(fill="both", expand=True, padx=16, pady=8)

    # ── Tab: Export ────────────────────────────────────────────────

    def _build_export_tab(self):
        f = self.tabs["export"]
        ttk.Label(f, text="EXPORT",
                  style="Heading.TLabel").pack(anchor="w", padx=16, pady=(16, 8))
        ttk.Button(f, text="Export Clean Dataset (CSV)",
                   command=lambda: self.controller.export("clean_csv")
                   ).pack(anchor="w", padx=16, pady=3)
        ttk.Button(f, text="Export Predictions (CSV)",
                   command=lambda: self.controller.export("pred_csv")
                   ).pack(anchor="w", padx=16, pady=3)
        ttk.Button(f, text="Export Risk Scores (CSV)",
                   command=lambda: self.controller.export("risk_csv")
                   ).pack(anchor="w", padx=16, pady=3)
        ttk.Button(f, text="Export Anomalies (CSV)",
                   command=lambda: self.controller.export("anomaly_csv")
                   ).pack(anchor="w", padx=16, pady=3)
        ttk.Button(f, text="Export Model Metrics (JSON)",
                   command=lambda: self.controller.export("metrics_json")
                   ).pack(anchor="w", padx=16, pady=3)
        ttk.Button(f, text="Export All Charts (PNG)",
                   command=lambda: self.controller.export("charts_png")
                   ).pack(anchor="w", padx=16, pady=3)
        ttk.Button(f, text="Export Summary (CSV + JSON)",
                   command=lambda: self.controller.export("summary")
                   ).pack(anchor="w", padx=16, pady=3)

        self.export_log = tk.Text(f, bg=COLORS["bg_mid"],
                                  fg=COLORS["fg_primary"],
                                  font=FONTS["mono"],
                                  relief="flat", wrap="word", height=10)
        self.export_log.pack(fill="both", expand=True, padx=16, pady=8)

    # ── Run ────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()