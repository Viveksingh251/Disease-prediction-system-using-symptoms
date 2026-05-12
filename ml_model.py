from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from diseases import Disease


@dataclass(frozen=True)
class VectorizerSpec:
    vocabulary: List[str]


def build_vocabulary(disease_db: Dict[str, Disease]) -> List[str]:
    vocab = set()
    for d in disease_db.values():
        for s in d.symptoms:
            vocab.add(s)
    # deterministic ordering
    return sorted(vocab)


def vectorize_symptoms(user_symptoms: List[str], vocabulary: List[str]) -> List[int]:
    vocab_index = {s: i for i, s in enumerate(vocabulary)}
    x = [0] * len(vocabulary)
    for s in user_symptoms:
        i = vocab_index.get(s)
        if i is not None:
            x[i] = 1
    return x


def build_synthetic_dataset(
    disease_db: Dict[str, Disease],
    *,
    max_subsets_per_disease: int = 30,
    min_subset_size: int = 1,
) -> Tuple[List[List[int]], List[str], List[str]]:
    """Create a tiny training set from the existing DISEASE_DB.

    Because this project currently has no real labeled patient dataset,
    we generate synthetic samples by taking random subsets of each
    disease's symptom list and labeling them with that disease.

    Returns:
      X (list of multi-hot vectors), y (list of disease keys), vocabulary
    """

    import random

    vocabulary = build_vocabulary(disease_db)

    keys = list(disease_db.keys())
    X: List[List[int]] = []
    y: List[str] = []

    rng = random.Random(42)

    for key in keys:
        symptoms = list(disease_db[key].symptoms)
        if not symptoms:
            continue

        # Always include the full symptom set to anchor the class.
        X.append(vectorize_symptoms(symptoms, vocabulary))
        y.append(key)

        # Generate additional subsets.
        subsets_generated = 0
        # Shuffle to vary subset composition.
        shuffled = symptoms[:]
        rng.shuffle(shuffled)
        while subsets_generated < max_subsets_per_disease:
            # pick a subset size
            k = rng.randint(min_subset_size, len(shuffled))
            subset = rng.sample(shuffled, k)
            if not subset:
                continue
            X.append(vectorize_symptoms(subset, vocabulary))
            y.append(key)
            subsets_generated += 1

    return X, y, vocabulary

