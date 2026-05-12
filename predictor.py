from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from diseases import Disease, normalize_symptom
from model_store import ModelArtifacts, get_artifacts, set_artifacts
from ml_model import build_synthetic_dataset, build_vocabulary, vectorize_symptoms


@dataclass(frozen=True)
class Prediction:
    disease: str
    score: float
    matched_symptoms: List[str]


# Model selection: Random Forest is generally more stable than a single Decision Tree.
ML_MODEL_TYPE = "random_forest"  # or "decision_tree"


def _train_and_cache_model(disease_db: Dict[str, Disease]):
    """Train a small classifier from DISEASE_DB.

    Note: This project has no real labeled patient dataset; we create a tiny
    synthetic training set from each disease's symptom list.
    """

    artifacts = get_artifacts()
    if artifacts is not None:
        return artifacts

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    X_list, y_keys, vocabulary = build_synthetic_dataset(disease_db)
    if not X_list:
        # fallback empty
        return None

    import numpy as np

    X = np.array(X_list, dtype=np.int64)
    label_map = sorted(set(y_keys))
    y = np.array([label_map.index(k) for k in y_keys], dtype=np.int64)

    if ML_MODEL_TYPE == "decision_tree":
        clf = DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=1)
    else:
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=None,
            min_samples_leaf=1,
        )

    clf.fit(X, y)

    artifacts = ModelArtifacts(model=clf, vocabulary=vocabulary, label_map=label_map)
    set_artifacts(artifacts)
    return artifacts


def predict_disease(
    user_symptoms: List[str],
    disease_db: Dict[str, Disease],
    top_k: int = 1,
) -> Prediction:
    user_symptoms = [normalize_symptom(s) for s in user_symptoms if normalize_symptom(s)]

    artifacts = _train_and_cache_model(disease_db)
    if artifacts is None:
        # ultimate fallback: choose the first disease
        first = next(iter(disease_db.values()))
        matched = sorted(set(user_symptoms) & set(first.symptoms))
        return Prediction(disease=first.name, score=0.0, matched_symptoms=matched)

    x_vec = vectorize_symptoms(user_symptoms, artifacts.vocabulary)

    import numpy as np

    Xq = np.array([x_vec], dtype=np.int64)

    clf = artifacts.model

    matched = None
    # compute matched symptoms based on predicted disease symptom list

    # Use predict_proba if available.
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Xq)[0]
        best_idx = int(np.argmax(proba))
        score = float(proba[best_idx])
    else:
        best_idx = int(clf.predict(Xq)[0])
        score = 0.0

    disease_key = artifacts.label_map[best_idx]
    disease = disease_db[disease_key]

    matched = sorted(set(user_symptoms) & set(disease.symptoms))
    return Prediction(disease=disease.name, score=score, matched_symptoms=matched)


