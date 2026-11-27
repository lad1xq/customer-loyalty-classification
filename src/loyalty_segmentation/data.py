"""Leakage-safe preprocessing for the retail dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreparedData:
    x_train: NDArray[np.float64]
    x_test: NDArray[np.float64]
    y_train: NDArray
    y_test: NDArray
    preprocessor: ColumnTransformer


def prepare_data(
    frame: pd.DataFrame,
    *,
    target_column: str = "customer_segment",
    id_column: str = "customer_id",
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreparedData:
    """Split first, then fit every preprocessing step on training data only."""
    required = {target_column, id_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    features = frame.drop(columns=[id_column, target_column])
    targets = frame[target_column].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )

    numeric_columns = x_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = x_train.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    transformed_train = preprocessor.fit_transform(x_train)
    transformed_test = preprocessor.transform(x_test)

    return PreparedData(
        x_train=np.asarray(transformed_train, dtype=float),
        x_test=np.asarray(transformed_test, dtype=float),
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
    )

