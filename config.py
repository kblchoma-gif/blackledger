"""BLACKLEDGER configuration and constants."""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
EXPORT_DIR = BASE_DIR / "exports"
LOG_DIR = BASE_DIR / "logs"
DB_PATH = BASE_DIR / "blackledger.db"

for _dir in (DATA_DIR, MODEL_DIR, EXPORT_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Random seed for reproducibility ────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ── Wednesday Addams palette ───────────────────────────────────────
COLORS = {
    "bg_darkest": "#0A0A0A",
    "bg_dark": "#121212",
    "bg_mid": "#1A1A1A",
    "bg_light": "#242424",
    "fg_primary": "#E8E8E8",
    "fg_secondary": "#A0A0A0",
    "fg_muted": "#6B6B6B",
    "accent": "#8B0000",        # dark red
    "accent_hover": "#B22222",
    "accent_soft": "#5C0000",
    "success": "#2E7D32",
    "warning": "#B8860B",
    "danger": "#C62828",
    "info": "#4A4A8A",
    "border": "#2C2C2C",
}

FONTS = {
    "title": ("Courier New", 20, "bold"),
    "heading": ("Courier New", 14, "bold"),
    "subheading": ("Courier New", 12, "bold"),
    "body": ("Courier New", 10),
    "mono": ("Courier New", 10),
    "small": ("Courier New", 9),
}

# ── Thresholds ─────────────────────────────────────────────────────
ANOMALY_CONTAMINATION = 0.05   # expected proportion of anomalies
RISK_HIGH_THRESHOLD = 0.70
RISK_MEDIUM_THRESHOLD = 0.40