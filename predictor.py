from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from diseases import Disease, normalize_symptom


@dataclass(frozen=True)
class Prediction:
    disease: str
    score: float
    matched_symptoms: List[str]


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def predict_disease(user_symptoms: List[str], disease_db: Dict[str, Disease], top_k: int = 1) -> Prediction:
    user_symptoms = [normalize_symptom(s) for s in user_symptoms if normalize_symptom(s)]

    best = None
    for key, d in disease_db.items():
        matched = sorted(set(user_symptoms) & set(d.symptoms))
        score = jaccard(user_symptoms, d.symptoms)

        if best is None or score > best.score:
            best = Prediction(disease=d.name, score=score, matched_symptoms=matched)

    assert best is not None
    return best

