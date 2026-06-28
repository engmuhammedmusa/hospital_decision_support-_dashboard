"""Decision support page for system alerts and patient interventions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.alerts import AlertEngine
from src.pipeline import get_dashboard_data
from src.styles import load_css


st.set_page_config(page_title="Decision Support | Hospital DSS", layout="wide", page_icon="H")
load_css()

df, metrics, kpis = get_dashboard_data()
st.title("Decision Support")
st.caption("Clinical alerts, intervention prioritization, and research disclaimer")

if df.empty:
    st.error("Clinical data is unavailable. Check the dataset path and reload the app.")
    st.stop()

st.subheader("System Alerts")
alerts = AlertEngine(df, kpis).generate_alerts()
for alert in alerts:
    message = alert["message"]
    if alert["type"] == "critical":
        st.error(f"[CRITICAL] {message}")
    elif alert["type"] == "warning":
        st.warning(f"[WARNING] {message}")
    elif alert["type"] == "success":
        st.success(f"[SUCCESS] {message}")
    else:
        st.info(message)

st.subheader("High Risk Patient Interventions")
intervention_df = df.sort_values("risk_score", ascending=False).head(10).copy()

def recommended_action(score: float) -> str:
    """Map risk score ranges to recommended intervention actions."""
    if score >= 80:
        return "Immediate clinical review required"
    if score >= 70:
        return "Schedule follow-up before discharge"
    if score >= 40:
        return "Monitor closely - daily check"
    return "Routine follow-up"


intervention_df["recommended_action"] = intervention_df["risk_score"].apply(recommended_action)
columns = ["patient_nbr", "age", "gender", "time_in_hospital", "risk_score", "risk_category", "recommended_action"]
styled = intervention_df[columns].style.bar(
    subset=["risk_score"],
    color="#1E88E5",
    vmin=0,
    vmax=100,
).format({"risk_score": "{:.1f}"})
st.dataframe(styled, width="stretch", hide_index=True)

st.subheader("System Info")
with st.expander("What is this system?"):
    st.write(
        "This dashboard is a research prototype that combines hospital operations analytics, "
        "readmission risk prediction, and transparent rule-based alerts."
    )
with st.expander("How is risk score calculated?"):
    st.write(
        "A Random Forest classifier estimates the probability of readmission within 30 days "
        "using age, length of stay, procedures, medications, diagnoses, and recent utilization."
    )
with st.expander("What does readmission risk mean?"):
    st.write(
        "Readmission risk is the estimated likelihood that a patient returns to the hospital "
        "within 30 days. It supports prioritization and is not a diagnosis."
    )
with st.expander("MDR/SaMD disclaimer"):
    st.warning(
        "This system is a prototype for research purposes. Clinical decisions must be made "
        "by qualified healthcare professionals."
    )
