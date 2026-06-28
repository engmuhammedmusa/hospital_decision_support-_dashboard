"""KPI calculation helpers for hospital operations and readmission risk."""

from __future__ import annotations

import pandas as pd


class KPICalculator:
    """Compute reusable dashboard KPIs from a scored clinical DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Store the DataFrame used by all KPI methods."""
        self.df = df

    def calculate(self) -> dict[str, object]:
        """Return all KPI values needed by the dashboard pages."""
        if self.df.empty:
            return {}

        high_risk_mask = self.df["risk_category"].eq("High Risk")
        readmitted_mask = self.df["readmitted_binary"].eq(1)

        return {
            "total_patients": int(len(self.df)),
            "avg_time_in_hospital": float(self.df["time_in_hospital"].mean()),
            "high_risk_count": int(high_risk_mask.sum()),
            "high_risk_rate": float(high_risk_mask.mean() * 100),
            "readmission_rate": float(readmitted_mask.mean() * 100),
            "avg_medications": float(self.df["num_medications"].mean()),
            "avg_lab_procedures": float(self.df["num_lab_procedures"].mean()),
            "top_diagnoses": self.df["diag_1"].value_counts().head(5),
        }
