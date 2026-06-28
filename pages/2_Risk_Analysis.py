"""Interactive patient-level readmission risk analysis page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import get_dashboard_data
from src.styles import load_css


st.set_page_config(page_title="Risk Analysis | Hospital DSS", layout="wide", page_icon="H")
load_css()

df, metrics, kpis = get_dashboard_data()
st.title("Patient Risk Analysis")
st.caption("Risk scoring, patient filtering, and model performance review")

if df.empty:
    st.error("Clinical data is unavailable. Check the dataset path and reload the app.")
    st.stop()

risk_options = ["High Risk", "Medium Risk", "Low Risk"]
selected_risks = st.sidebar.multiselect("Risk category", risk_options, default=risk_options)
age_min, age_max = int(df["age"].min()), int(df["age"].max())
selected_age = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
stay_min, stay_max = int(df["time_in_hospital"].min()), int(df["time_in_hospital"].max())
selected_stay = st.sidebar.slider("Time in hospital", stay_min, stay_max, (stay_min, stay_max))

filtered_df = df[
    df["risk_category"].isin(selected_risks)
    & df["age"].between(selected_age[0], selected_age[1])
    & df["time_in_hospital"].between(selected_stay[0], selected_stay[1])
].copy()

display_columns = [
    "patient_nbr",
    "age",
    "gender",
    "time_in_hospital",
    "num_medications",
    "number_diagnoses",
    "risk_score",
    "risk_category",
]
top_patients = filtered_df.sort_values("risk_score", ascending=False).head(50)[display_columns]

def highlight_risk(row: pd.Series) -> list[str]:
    """Color table rows based on the model risk category."""
    color = {
        "High Risk": "background-color: #FDECEA",
        "Medium Risk": "background-color: #FFF4E5",
        "Low Risk": "background-color: #ECF7ED",
    }.get(row["risk_category"], "")
    return [color] * len(row)


st.subheader("Top 50 Patients by Risk Score")
st.dataframe(
    top_patients.style.apply(highlight_risk, axis=1).format({"risk_score": "{:.1f}"}),
    width="stretch",
    hide_index=True,
)

left, right = st.columns(2)
with left:
    fig = px.scatter(
        filtered_df,
        x="time_in_hospital",
        y="risk_score",
        color="risk_category",
        color_discrete_map={"High Risk": "#E53935", "Medium Risk": "#FF6F00", "Low Risk": "#43A047"},
        title="Risk vs Stay",
        labels={"time_in_hospital": "Time in Hospital (days)", "risk_score": "Risk Score"},
        template="plotly_white",
    )
    st.plotly_chart(fig, width="stretch")

with right:
    age_risk = filtered_df.groupby("age_group", observed=False)["risk_score"].mean().reset_index()
    fig = px.bar(
        age_risk,
        x="age_group",
        y="risk_score",
        title="Avg Risk by Age",
        labels={"age_group": "Age Group", "risk_score": "Average Risk Score"},
        color_discrete_sequence=["#1E88E5"],
        template="plotly_white",
    )
    st.plotly_chart(fig, width="stretch")

st.subheader("Model Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
m2.metric("Precision", f"{metrics.get('precision', 0):.2%}")
m3.metric("Recall", f"{metrics.get('recall', 0):.2%}")
m4.metric("F1 Score", f"{metrics.get('f1', 0):.2%}")
st.markdown(
    """
    <p class="small-note">
    <strong>Why not just accuracy?</strong> Clinical datasets can be imbalanced.
    A model may look accurate while missing patients who are readmitted within 30 days.
    Recall is especially important because missed high-risk patients can reduce the
    usefulness of a decision support workflow.
    </p>
    """,
    unsafe_allow_html=True,
)
