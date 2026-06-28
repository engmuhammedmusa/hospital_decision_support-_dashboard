"""Deep analytics page for demographic, clinical, and readmission patterns."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.pipeline import get_dashboard_data
from src.styles import load_css


st.set_page_config(page_title="Analytics | Hospital DSS", layout="wide", page_icon="H")
load_css()

df, metrics, kpis = get_dashboard_data()
st.title("Clinical Analytics")
st.caption("Population-level demographics, utilization patterns, and readmission analysis")

if df.empty:
    st.error("Clinical data is unavailable. Check the dataset path and reload the app.")
    st.stop()

st.subheader("Patient Demographics")
demo_1, demo_2, demo_3 = st.columns(3)
with demo_1:
    gender_counts = df["gender"].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Patients"]
    fig = px.bar(gender_counts, x="Gender", y="Patients", title="Gender Distribution", template="plotly_white")
    fig.update_traces(marker_color="#1E88E5")
    fig.update_layout(xaxis_title="Gender", yaxis_title="Patient Count")
    st.plotly_chart(fig, width="stretch")
with demo_2:
    age_counts = df["age_group"].value_counts().sort_index().reset_index()
    age_counts.columns = ["Age Group", "Patients"]
    fig = px.bar(age_counts, x="Age Group", y="Patients", title="Age Group Distribution", template="plotly_white")
    fig.update_traces(marker_color="#43A047")
    fig.update_layout(xaxis_title="Age Group", yaxis_title="Patient Count")
    st.plotly_chart(fig, width="stretch")
with demo_3:
    race_counts = df["race"].fillna("Unknown").value_counts().reset_index()
    race_counts.columns = ["Race", "Patients"]
    fig = px.bar(race_counts, x="Race", y="Patients", title="Race Distribution", template="plotly_white")
    fig.update_traces(marker_color="#FF6F00")
    fig.update_layout(xaxis_title="Race", yaxis_title="Patient Count")
    st.plotly_chart(fig, width="stretch")

st.subheader("Clinical Patterns")
clin_1, clin_2, clin_3 = st.columns(3)
with clin_1:
    fig = px.histogram(
        df,
        x="num_medications",
        nbins=35,
        title="Number of Medications",
        labels={"num_medications": "Number of Medications"},
        template="plotly_white",
    )
    fig.update_traces(marker_color="#1E88E5")
    st.plotly_chart(fig, width="stretch")
with clin_2:
    fig = px.histogram(
        df,
        x="num_lab_procedures",
        nbins=35,
        title="Number of Lab Procedures",
        labels={"num_lab_procedures": "Number of Lab Procedures"},
        template="plotly_white",
    )
    fig.update_traces(marker_color="#43A047")
    st.plotly_chart(fig, width="stretch")
with clin_3:
    fig = px.scatter(
        df.sample(min(len(df), 5000), random_state=42),
        x="num_medications",
        y="time_in_hospital",
        color="risk_category",
        color_discrete_map={"High Risk": "#E53935", "Medium Risk": "#FF6F00", "Low Risk": "#43A047"},
        title="Medications vs Time in Hospital",
        labels={"num_medications": "Number of Medications", "time_in_hospital": "Time in Hospital (days)"},
        template="plotly_white",
    )
    st.plotly_chart(fig, width="stretch")

st.subheader("Readmission Analysis")
read_1, read_2 = st.columns(2)
with read_1:
    age_readmission = df.groupby("age_group", observed=False)["readmitted_binary"].mean().mul(100).reset_index()
    fig = px.bar(
        age_readmission,
        x="age_group",
        y="readmitted_binary",
        title="Readmission Rate by Age Group",
        labels={"age_group": "Age Group", "readmitted_binary": "Readmission Rate (%)"},
        template="plotly_white",
    )
    fig.update_traces(marker_color="#E53935")
    st.plotly_chart(fig, width="stretch")
with read_2:
    diagnosis_readmission = (
        df.groupby("number_diagnoses")["readmitted_binary"].mean().mul(100).reset_index()
    )
    fig = px.bar(
        diagnosis_readmission,
        x="number_diagnoses",
        y="readmitted_binary",
        title="Readmission Rate by Number of Diagnoses",
        labels={"number_diagnoses": "Number of Diagnoses", "readmitted_binary": "Readmission Rate (%)"},
        template="plotly_white",
    )
    fig.update_traces(marker_color="#FF6F00")
    st.plotly_chart(fig, width="stretch")

numeric_features = [
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
    "risk_score",
]
corr = df[numeric_features].corr()
fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    title="Correlation Matrix of Numerical Features",
    template="plotly_white",
    aspect="auto",
)
fig.update_layout(xaxis_title="Feature", yaxis_title="Feature")
st.plotly_chart(fig, width="stretch")
