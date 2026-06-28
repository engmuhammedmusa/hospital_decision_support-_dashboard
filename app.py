"""Main Streamlit entry point for the Hospital Decision Support Dashboard."""

from __future__ import annotations

from datetime import datetime

import plotly.express as px
import streamlit as st

from src.pipeline import get_dashboard_data
from src.styles import load_css


st.set_page_config(
    page_title="Hospital DSS",
    layout="wide",
    page_icon="H",
    initial_sidebar_state="expanded",
)
load_css()


def render_sidebar(model_loaded: bool) -> None:
    """Render shared sidebar context and model status."""
    st.sidebar.markdown("## Hospital DSS")
    st.sidebar.caption("Clinical decision support prototype")
    st.sidebar.markdown("Use the pages menu to review operations, risk analysis, analytics, and alerts.")
    st.sidebar.divider()
    st.sidebar.markdown("**Dataset**")
    st.sidebar.caption("Source: Diabetes 130-US Hospitals Dataset, UCI ML Repository")
    st.sidebar.markdown(
        f"<span class='status-dot'></span>{'Model loaded' if model_loaded else 'Model unavailable'}",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.markdown("[By Muhammed MUSA](https://msm-bio.vercel.app/)")


df, metrics, kpis = get_dashboard_data()
render_sidebar(model_loaded=bool(metrics))

st.markdown(
    """
    <div class="hero-panel">
      <h1>Hospital Decision Support Dashboard</h1>
      <p class="small-note">
        A Streamlit system for clinical operations analytics,
        30-day readmission risk scoring, and rule-based care recommendations.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if df.empty:
    st.error("The dashboard could not load the dataset. Confirm that data/diabetic_data.csv exists.")
    st.stop()

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

stat_1, stat_2, stat_3, stat_4 = st.columns(4)
stat_1.metric("Total Patients", f"{kpis['total_patients']:,}")
stat_2.metric("Avg Stay", f"{kpis['avg_time_in_hospital']:.1f} days")
stat_3.metric("High Risk", f"{kpis['high_risk_count']:,}", f"{kpis['high_risk_rate']:.1f}%")
stat_4.metric("Readmission Rate", f"{kpis['readmission_rate']:.1f}%")

st.subheader("Getting Started")
st.write(
    "Open the pages in the sidebar to explore hospital operations, patient-level risk, "
    "population analytics, and decision support alerts. The model is trained on startup "
    "using cleaned clinical utilization and demographic features."
)

left, right = st.columns(2)
with left:
    risk_counts = df["risk_category"].value_counts().reset_index()
    risk_counts.columns = ["Risk Category", "Patients"]
    fig = px.bar(
        risk_counts,
        x="Risk Category",
        y="Patients",
        color="Risk Category",
        color_discrete_map={"High Risk": "#E53935", "Medium Risk": "#FF6F00", "Low Risk": "#43A047"},
        title="Current Risk Distribution",
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, xaxis_title="Risk Category", yaxis_title="Patient Count")
    st.plotly_chart(fig, width="stretch")

with right:
    metric_df = (
        df.groupby("age_group", observed=False)["readmitted_binary"].mean().mul(100).reset_index(name="Readmission Rate")
    )
    fig = px.line(
        metric_df,
        x="age_group",
        y="Readmission Rate",
        markers=True,
        title="30-Day Readmission Rate by Age Group",
        template="plotly_white",
    )
    fig.update_traces(line_color="#1E88E5")
    fig.update_layout(xaxis_title="Age Group", yaxis_title="Readmission Rate (%)")
    st.plotly_chart(fig, width="stretch")
