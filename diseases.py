from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


def normalize_symptom(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("-", " ")
    # light normalization
    synonyms = {
        "high fever": "fever",
        "temperature": "fever",
        "sore throat": "throat pain",
    }
    return synonyms.get(s, s)


@dataclass(frozen=True)
class Disease:
    name: str
    symptoms: List[str]
    treatments: Dict[str, List[str]]
    # Optional extra hints
    contraindications: List[str]


# Minimal baseline knowledge base (extend freely)
# Treatments are grouped by template ids.
DISEASE_DB: Dict[str, Disease] = {
    "flu": Disease(
        name="Influenza (Flu)",
        symptoms=[
            "fever",
            "cough",
            "body aches",
            "headache",
            "fatigue",
            "chills",
        ],
        treatments={
            "conservative": [
                "Rest and hydration",
                "Paracetamol/acetaminophen for fever and pain (follow label) ",
                "Warm fluids, honey for cough (if age-appropriate)",
                "Consider medical evaluation if symptoms are severe or persistent",
            ],
            "balanced": [
                "Everything in conservative tier",
                "Antiviral therapy (e.g., oseltamivir) may help if started early—discuss with a clinician",
                "Monitor for complications (breathing difficulty, dehydration)",
            ],
            "aggressive": [
                "Everything in balanced tier",
                "Early clinician assessment; consider diagnostic testing if high-risk patient",
            ],
        },
        contraindications=["aspirin allergy"],
    ),
    "common_cold": Disease(
        name="Common Cold",
        symptoms=[
            "cough",
            "throat pain",
            "runny nose",
            "sneezing",
            "fatigue",
            "headache",
        ],
        treatments={
            "conservative": [
                "Rest and fluids",
                "Saline nasal rinse/spray",
                "Throat lozenges; warm salt-water gargles",
                "Acetaminophen/ibuprofen for pain/fever if appropriate",
            ],
            "balanced": [
                "Everything in conservative tier",
                "Symptomatic relief: decongestant or cough suppressant only if suitable",
                "Seek care if fever lasts >3 days, or symptoms worsen",
            ],
            "aggressive": [
                "Everything in balanced tier",
                "Medical evaluation to rule out sinus infection/bronchitis if persistent.",
            ],
        },
        contraindications=[],
    ),
    "migraine": Disease(
        name="Migraine",
        symptoms=[
            "headache",
            "nausea",
            "sensitivity to light",
            "aura",
            "fatigue",
        ],
        treatments={
            "conservative": [
                "Hydration and rest in a dark room",
                "Cold compress on forehead",
                "Pain relief meds (e.g., acetaminophen) if appropriate",
                "Keep a symptom/trigger diary",
            ],
            "balanced": [
                "Everything in conservative tier",
                "Consider migraine-specific meds (tripans) if prescribed",
                "Anti-nausea support if needed—discuss with clinician",
            ],
            "aggressive": [
                "Everything in balanced tier",
                "Prompt clinical review for frequent/severe attacks or unusual symptoms",
            ],
        },
        contraindications=[],
    ),
}

