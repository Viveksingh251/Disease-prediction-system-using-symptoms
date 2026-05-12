"""Disease Prediction System

End-to-end demo:
- Predict disease from symptoms (baseline: Jaccard similarity)
- Recommend personalized treatment (personalities/strategies)

Run: python app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from diseases import DISEASE_DB, normalize_symptom
from predictor import predict_disease
from recommenders import build_personalities, recommend_treatment


@dataclass
class PatientProfile:
    age: int
    sex: str  # "male" | "female" | "other"
    allergies: List[str]
    conditions: List[str]
    risk_tolerance: str  # "low" | "medium" | "high"


def parse_list_csv(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    print("Disease Prediction System (Symptoms -> Disease + Treatment Recommendations)\n")

    print("Enter patient details:")
    age = int(input("Age (e.g., 25): ").strip() or "0")
    sex = input("Sex (male/female/other) [other]: ").strip().lower() or "other"
    allergies = parse_list_csv(input("Allergies (comma-separated, optional): ").strip())
    conditions = parse_list_csv(input("Existing conditions (comma-separated, optional): ").strip())
    risk_tolerance = input("Risk tolerance (low/medium/high) [medium]: ").strip().lower() or "medium"

    print("\nEnter symptoms (comma-separated). Examples: fever, cough, headache")
    raw_symptoms = input("Symptoms: ").strip()
    user_symptoms = [normalize_symptom(x) for x in raw_symptoms.split(",") if normalize_symptom(x)]

    if not user_symptoms:
        print("No symptoms provided. Exiting.")
        return

    profile = PatientProfile(
        age=age,
        sex=sex,
        allergies=[normalize_symptom(a) for a in allergies],
        conditions=[normalize_symptom(c) for c in conditions],
        risk_tolerance=risk_tolerance,
    )

    print("\nPredicting disease...")
    pred = predict_disease(user_symptoms, DISEASE_DB)

    print("\nTop prediction:")
    print(f"- Disease: {pred.disease}")
    print(f"- Score: {pred.score:.3f}")
    print(f"- Matching symptoms: {', '.join(pred.matched_symptoms) or 'None'}")

    personalities = build_personalities()

    print("\nTreatment recommendation (personalized):")
    # Choose which personality to use based on user risk tolerance
    personality_key = "conservative" if profile.risk_tolerance == "low" else "balanced" if profile.risk_tolerance == "medium" else "aggressive"

    personality = personalities[personality_key]
    treatment = recommend_treatment(pred.disease, profile, personality)

    print(treatment)

    # Show extensibility: add additional personalities
    print("\nOptional: other personalities (you can add more) ")
    for key, p in personalities.items():
        if key == personality_key:
            continue
        t2 = recommend_treatment(pred.disease, profile, p)
        print(f"\n--- {p.name.upper()} ---\n{t2}")

    print("\nDISCLAIMER: This is a demo and not medical advice. Consult a professional.")


if __name__ == "__main__":
    main()

