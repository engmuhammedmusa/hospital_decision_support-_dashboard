"""Data loading and cleaning utilities for the Hospital DSS dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


class DataLoader:
    """Load and clean the Diabetes 130-US Hospitals dataset."""

    def __init__(self, data_path: Path | None = None) -> None:
        """Initialize the loader with a pathlib-based CSV path."""
        self.data_path = data_path or Path(__file__).resolve().parents[1] / "data" / "diabetic_data.csv"

    @staticmethod
    def _age_to_midpoint(age_bracket: str) -> int | float:
        """Convert an age bracket such as ``[70-80)`` into its midpoint."""
        if not isinstance(age_bracket, str) or "-" not in age_bracket:
            return np.nan

        cleaned = age_bracket.strip("[]()")
        lower, upper = cleaned.split("-")
        return int((int(lower) + int(upper)) / 2)

    @staticmethod
    @st.cache_data(show_spinner="Loading and cleaning clinical dataset...")
    def _load_cached(data_path: str) -> pd.DataFrame:
        """Load the CSV once and cache the cleaned DataFrame for Streamlit reruns."""
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")

        df = pd.read_csv(path)
        df = df.replace("?", np.nan)

        # Drop sparse columns so downstream analytics stay reliable.
        missing_ratio = df.isna().mean()
        sparse_columns = missing_ratio[missing_ratio > 0.40].index
        df = df.drop(columns=sparse_columns)

        if "age" in df.columns:
            df["age_group"] = df["age"]
            df["age"] = df["age"].apply(DataLoader._age_to_midpoint)

        risk_map = {"<30": "High Risk", ">30": "Medium Risk", "NO": "Low Risk"}
        df["readmission_risk"] = df["readmitted"].map(risk_map).fillna("Unknown")
        df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)

        numeric_columns = [
            "age",
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_diagnoses",
            "number_inpatient",
            "number_emergency",
            "number_outpatient",
        ]
        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
                df[column] = df[column].fillna(df[column].median())

        if "gender" in df.columns:
            df = df[df["gender"].isin(["Male", "Female"])].copy()

        return df.reset_index(drop=True)

    def load_data(self) -> pd.DataFrame:
        """Return a cleaned DataFrame, surfacing useful errors in the app."""
        try:
            return self._load_cached(str(self.data_path))
        except Exception as exc:
            st.error(f"Unable to load clinical dataset: {exc}")
            return pd.DataFrame()
