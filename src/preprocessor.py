"""Feature engineering utilities for the readmission risk model."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """Prepare model features and target labels."""

    feature_columns = [
        "age",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_diagnoses",
        "number_inpatient",
        "number_emergency",
        "number_outpatient",
        "gender_binary",
    ]

    def __init__(self) -> None:
        """Create a reusable StandardScaler instance."""
        self.scaler = StandardScaler()

    def prepare_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Encode, scale, and return model-ready X and y."""
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=int)

        model_df = df.copy()
        model_df["gender_binary"] = model_df["gender"].map({"Male": 1, "Female": 0}).fillna(0)

        available_features = [column for column in self.feature_columns if column in model_df.columns]
        X = model_df[available_features].copy()
        y = model_df["readmitted_binary"].astype(int)

        X = X.fillna(X.median(numeric_only=True))
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=available_features,
            index=X.index,
        )
        return X_scaled, y
