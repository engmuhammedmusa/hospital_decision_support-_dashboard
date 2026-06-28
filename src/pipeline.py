"""Shared application pipeline for loading data, scoring risk, and computing KPIs."""

from __future__ import annotations

import pandas as pd

from src.data_loader import DataLoader
from src.kpi import KPICalculator
from src.model import RiskModel


def get_dashboard_data() -> tuple[pd.DataFrame, dict[str, float], dict[str, object]]:
    """Return scored patient data, model metrics, and operational KPIs."""
    loader = DataLoader()
    df = loader.load_data()
    if df.empty:
        return df, {}, {}

    risk_model = RiskModel()
    scored_df, metrics = risk_model.fit_predict(df)
    kpis = KPICalculator(scored_df).calculate()
    return scored_df, metrics, kpis
