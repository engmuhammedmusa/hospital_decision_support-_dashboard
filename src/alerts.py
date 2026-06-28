"""Clinical decision support alert logic."""

from __future__ import annotations

import pandas as pd


class AlertEngine:
    """Generate rule-based and patient-level decision support alerts."""

    def __init__(self, df: pd.DataFrame, kpis: dict[str, object]) -> None:
        """Store scored patient data and precomputed KPI values."""
        self.df = df
        self.kpis = kpis

    def generate_alerts(self) -> list[dict[str, str]]:
        """Return system-level and patient-level alerts as dictionaries."""
        alerts: list[dict[str, str]] = []
        if self.df.empty or not self.kpis:
            return [{"type": "warning", "message": "Clinical data is unavailable. Check dataset loading."}]

        if self.kpis.get("high_risk_rate", 0) > 15:
            alerts.append(
                {
                    "type": "warning",
                    "message": "High readmission risk rate detected. Review discharge protocols.",
                }
            )

        if self.kpis.get("avg_time_in_hospital", 0) > 5:
            alerts.append(
                {
                    "type": "warning",
                    "message": "Average hospital stay exceeds 5 days. Investigate bottlenecks.",
                }
            )

        age_risk = (
            self.df.groupby("age_group", observed=False)["risk_category"]
            .apply(lambda values: values.eq("High Risk").mean() * 100)
            .sort_values(ascending=False)
        )
        critical_age_groups = age_risk[age_risk > 30]
        for age_group, risk_rate in critical_age_groups.items():
            alerts.append(
                {
                    "type": "critical",
                    "message": f"Elderly patients ({age_group}) show critical readmission risk: {risk_rate:.1f}%.",
                }
            )

        if self.kpis.get("readmission_rate", 100) < 10:
            alerts.append(
                {
                    "type": "success",
                    "message": "Readmission rate is within acceptable range.",
                }
            )

        high_risk_patients = (
            self.df[self.df["risk_category"].eq("High Risk")]
            .sort_values("risk_score", ascending=False)
            .head(10)
        )
        for _, patient in high_risk_patients.iterrows():
            alerts.append(
                {
                    "type": "warning",
                    "message": (
                        f"Patient {patient['patient_nbr']} - Risk Score: {patient['risk_score']:.1f} - "
                        "Recommend pre-discharge intervention."
                    ),
                }
            )

        return alerts
