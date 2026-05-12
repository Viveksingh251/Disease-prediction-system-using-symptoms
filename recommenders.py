from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

from diseases import DISEASE_DB


class Personality(Protocol):
    key: str
    name: str

    def adjust_for_patient(self, base: List[str], profile: "PatientProfile") -> List[str]:
        ...


def _avoid_allergies(steps: List[str], allergies: List[str]) -> List[str]:
    if not allergies:
        return steps
    # naive keyword filtering
    lowered_all = [a.lower() for a in allergies]
    out: List[str] = []
    for step in steps:
        if any(a in step.lower() for a in lowered_all):
            continue
        out.append(step)
    return out


@dataclass(frozen=True)
class ConservativePersonality:
    key: str = "conservative"
    name: str = "conservative"

    def adjust_for_patient(self, base: List[str], profile: "PatientProfile") -> List[str]:
        steps = list(base)
        steps = _avoid_allergies(steps, profile.allergies)
        if profile.age > 65:
            steps.insert(0, "Extra caution for older adults: confirm dosing with a clinician/pharmacist.")
        return steps


@dataclass(frozen=True)
class BalancedPersonality:
    key: str = "balanced"
    name: str = "balanced"

    def adjust_for_patient(self, base: List[str], profile: "PatientProfile") -> List[str]:
        steps = list(base)
        steps = _avoid_allergies(steps, profile.allergies)
        if profile.conditions:
            steps.append("Because you have existing conditions, check interactions with your current medications.")
        return steps


@dataclass(frozen=True)
class AggressivePersonality:
    key: str = "aggressive"
    name: str = "aggressive"

    def adjust_for_patient(self, base: List[str], profile: "PatientProfile") -> List[str]:
        steps = list(base)
        steps = _avoid_allergies(steps, profile.allergies)
        if profile.age < 12:
            steps.insert(0, "For children, dosing and safety must be verified by a clinician.")
        return steps


def build_personalities() -> Dict[str, Personality]:
    return {
        "conservative": ConservativePersonality(),
        "balanced": BalancedPersonality(),
        "aggressive": AggressivePersonality(),
    }


def recommend_treatment(disease_name: str, profile: "PatientProfile", personality: Personality) -> str:
    # find disease by name
    disease = None
    for d in DISEASE_DB.values():
        if d.name == disease_name:
            disease = d
            break
    if disease is None:
        return "No treatment data found for this disease."

    base_steps = disease.treatments.get(personality.key, [])
    adjusted = personality.adjust_for_patient(base_steps, profile)

    lines = [
        f"Strategy/personality: {personality.name}",
        f"Personalized steps for {disease.name}:",
    ]
    if not adjusted:
        lines.append("- No specific steps available after personalization filters.")
    else:
        for i, s in enumerate(adjusted, 1):
            lines.append(f"{i}. {s}")

    lines.append("\nIf symptoms are severe, rapidly worsening, or you have urgent concerns, seek medical care.")
    return "\n".join(lines)

