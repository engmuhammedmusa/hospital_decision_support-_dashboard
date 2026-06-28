"""Machine learning model for hospital readmission risk scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.preprocessor import Preprocessor


class RiskModel:
    """Train a Random Forest model and attach risk predictions to patient rows."""

    def __init__(self) -> None:
        """Initialize an empty metrics dictionary."""
        self.metrics: dict[str, float] = {}

    @staticmethod
    @st.cache_resource(show_spinner="Training readmission risk model...")
    def _train_cached(X: pd.DataFrame, y: pd.Series) -> tuple[RandomForestClassifier, dict[str, float]]:
        """Train and cache the Random Forest classifier."""
        if X.empty or y.empty:
            raise ValueError("Cannot train model with empty features or target.")

        stratify = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }
        return model, metrics

    @staticmethod
    def categorize_score(score: float) -> str:
        """Translate a numeric risk score into a clinical risk category."""
        if score >= 70:
            return "High Risk"
        if score >= 40:
            return "Medium Risk"
        return "Low Risk"

    @staticmethod
    def category_color(category: str) -> str:
        """Return the dashboard color associated with a risk category."""
        return {
            "High Risk": "#E53935",
            "Medium Risk": "#FF6F00",
            "Low Risk": "#43A047",
        }.get(category, "#607D8B")

    def fit_predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        """Train the model and return patient rows with risk scores attached."""
        if df.empty:
            return df, {}

        preprocessor = Preprocessor()
        X, y = preprocessor.prepare_data(df)
        model, metrics = self._train_cached(X, y)
        self.metrics = metrics

        scored_df = df.copy()
        probabilities = model.predict_proba(X)
        positive_index = int(np.where(model.classes_ == 1)[0][0]) if 1 in model.classes_ else 0
        scored_df["risk_score"] = (probabilities[:, positive_index] * 100).round(1)
        scored_df["risk_category"] = scored_df["risk_score"].apply(self.categorize_score)
        scored_df["risk_color"] = scored_df["risk_category"].apply(self.category_color)
        return scored_df, metrics
