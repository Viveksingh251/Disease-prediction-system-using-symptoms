from __future__ import annotations


from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass(frozen=True)
class TrainingData:
    X: "pd.DataFrame"
    y: "pd.Series"
    feature_columns: List[str]
    label_map: List[str]


def _clean_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unused junk columns frequently seen in the dataset export.
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    drop_cols += ["fluid_overload.1"] if "fluid_overload.1" in df.columns else []
    df = df.drop(columns=drop_cols, errors="ignore")
    return df


def load_training_csv(csv_path: str) -> TrainingData:
    """Load Training.csv and return prepared X/y.

    Expects:
      - feature columns are all binary symptom columns
      - label column is `prognosis`

    Cleans known extra columns (e.g., `Unnamed: 133`, `fluid_overload.1`).
    """

    df = pd.read_csv(csv_path)
    df = _clean_training_columns(df)

    if "prognosis" not in df.columns:
        raise ValueError("Training.csv must contain a 'prognosis' column")

    # Feature columns = all except prognosis
    feature_columns = [c for c in df.columns if c != "prognosis"]

    # Ensure deterministic column order
    feature_columns = list(feature_columns)

    # Cast features to numeric (0/1)
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    y = df["prognosis"].astype(str)

    label_map = sorted(y.unique().tolist())

    return TrainingData(X=X, y=y, feature_columns=feature_columns, label_map=label_map)


def vectorize_user_symptoms(user_symptoms: List[str], feature_columns: List[str]) -> List[int]:
    col_index = {c: i for i, c in enumerate(feature_columns)}
    x = [0] * len(feature_columns)
    for s in user_symptoms:
        # Training.csv symptom strings are already normalized; keep exact match.
        i = col_index.get(s)
        if i is not None:
            x[i] = 1
    return x


def train_model_from_training_data(training: TrainingData, *, model_type: str = "random_forest") -> Tuple[object, List[str]]:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    label_to_idx = {k: i for i, k in enumerate(training.label_map)}
    y_idx = np.array([label_to_idx[k] for k in training.y.tolist()], dtype=np.int64)

    X_np = training.X.to_numpy(dtype=np.int64)

    if model_type == "decision_tree":
        clf = DecisionTreeClassifier(random_state=42, max_depth=12, min_samples_leaf=1)
    else:
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
        )

    clf.fit(X_np, y_idx)
    return clf, training.label_map


