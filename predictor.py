from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from diseases import Disease, normalize_symptom
from model_store import ModelArtifacts, get_artifacts, set_artifacts
from ml_model import load_training_csv, train_model_from_training_data, vectorize_user_symptoms



@dataclass(frozen=True)
class Prediction:
    disease: str
    score: float
    matched_symptoms: List[str]


# Model selection: Random Forest is generally more stable than a single Decision Tree.
ML_MODEL_TYPE = "random_forest"  # or "decision_tree"


def _train_and_cache_model(disease_db: Dict[str, Disease]):
    """Train from Training.csv and cache the fitted model artifacts."""

    artifacts = get_artifacts()
    if artifacts is not None:
        return artifacts

    # Training.csv is stored in ../Desktop/ relative to this project folder.
    training = load_training_csv(csv_path="../Desktop/Training.csv")
    clf, label_map = train_model_from_training_data(training, model_type=ML_MODEL_TYPE)

    artifacts = ModelArtifacts(
        model=clf,
        feature_columns=training.feature_columns,
        label_map=label_map,
    )
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

    x_vec = vectorize_user_symptoms(user_symptoms, artifacts.feature_columns)


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

    predicted_label = artifacts.label_map[best_idx]

    # Map predicted CSV label into our treatment knowledge base if possible.
    # If not found, still return the predicted label.
    disease = None
    for d in disease_db.values():
        if d.name == predicted_label or d.name.lower() == predicted_label.lower():
            disease = d
            break

    if disease is None:
        return Prediction(disease=predicted_label, score=score, matched_symptoms=sorted(set(user_symptoms)))

    matched = sorted(set(user_symptoms) & set(disease.symptoms))
    return Prediction(disease=disease.name, score=score, matched_symptoms=matched)



