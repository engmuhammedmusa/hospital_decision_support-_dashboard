"""Operations overview page for hospital KPIs and risk distribution."""

from __future__ import annotations

from datetime import datetime

import plotly.express as px
import streamlit as st

from src.pipeline import get_dashboard_data
from src.styles import load_css


st.set_page_config(page_title="Overview | Hospital DSS", layout="wide", page_icon="H")
load_css()

df, metrics, kpis = get_dashboard_data()
st.title("Hospital Operations Overview")
st.caption(f"Real-time Clinical Decision Support | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

if df.empty:
    st.error("Clinical data is unavailable. Check the dataset path and reload the app.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Patients", f"{kpis['total_patients']:,}")
col2.metric("Avg Stay (days)", f"{kpis['avg_time_in_hospital']:.1f}")
col3.metric("High Risk Patients", f"{kpis['high_risk_count']:,}", f"{kpis['high_risk_rate']:.1f}%")
col4.metric("Readmission Rate", f"{kpis['readmission_rate']:.1f}%")

left, right = st.columns(2)
with left:
    age_counts = df["age_group"].value_counts().sort_index().reset_index()
    age_counts.columns = ["Age Group", "Patients"]
    fig = px.bar(
        age_counts,
        x="Age Group",
        y="Patients",
        title="Patient Count by Age Group",
        color_discrete_sequence=["#1E88E5"],
        template="plotly_white",
    )
    fig.update_layout(xaxis_title="Age Group", yaxis_title="Patient Count")
    st.plotly_chart(fig, width="stretch")

with right:
    risk_counts = df["risk_category"].value_counts().reset_index()
    risk_counts.columns = ["Risk Category", "Patients"]
    fig = px.pie(
        risk_counts,
        names="Risk Category",
        values="Patients",
        title="Risk Category Distribution",
        color="Risk Category",
        color_discrete_map={"High Risk": "#E53935", "Medium Risk": "#FF6F00", "Low Risk": "#43A047"},
        template="plotly_white",
    )
    st.plotly_chart(fig, width="stretch")

histogram = df["time_in_hospital"].value_counts().sort_index().reset_index()
histogram.columns = ["Time in Hospital", "Patients"]
fig = px.line(
    histogram,
    x="Time in Hospital",
    y="Patients",
    markers=True,
    title="Time in Hospital Distribution",
    template="plotly_white",
)
fig.update_traces(line_color="#1E88E5")
fig.update_layout(xaxis_title="Time in Hospital (days)", yaxis_title="Patient Count")
st.plotly_chart(fig, width="stretch")
