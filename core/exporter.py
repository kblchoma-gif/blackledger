"""Export utilities: CSV, JSON, PNG."""
import json
import pandas as pd
from pathlib import Path
from typing import Optional
from config import EXPORT_DIR
from logger import get_logger

log = get_logger(__name__)

class Exporter:
    """Exports dataframes, metrics, and charts."""

    @staticmethod
    def to_csv(df: pd.DataFrame, name: str) -> Path:
        path = EXPORT_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        log.info("Exported CSV: %s", path)
        return path

    @staticmethod
    def to_json(data, name: str) -> Path:
        path = EXPORT_DIR / f"{name}.json"
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        log.info("Exported JSON: %s", path)
        return path

    @staticmethod
    def to_png(fig, name: str, dpi: int = 150) -> Path:
        path = EXPORT_DIR / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        log.info("Exported PNG: %s", path)
        return path

    @staticmethod
    def summary_report(df: pd.DataFrame, name: str = "summary") -> Path:
        """Export descriptive statistics as CSV + JSON."""
        desc = df.describe(include="all").transpose()
        path_csv = Exporter.to_csv(desc, f"{name}_describe")
        path_json = Exporter.to_json(
            desc.to_dict(), f"{name}_describe")
        return path_csv