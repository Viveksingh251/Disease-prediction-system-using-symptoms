from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib

# Use non-interactive backend for servers/CLI environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


@dataclass(frozen=True)
class EvalOutputs:
    confusion_matrix_png: bytes
    accuracy_curve_png: bytes
    score_curve_png: bytes


def _normalize_label_order(label_map: List[str]) -> List[str]:
    # deterministic order
    return list(label_map)


def evaluate_classifier(
    *,
    y_true: List[str],
    y_pred: List[str],
    predicted_scores: List[float],
    label_map: List[str],
) -> EvalOutputs:
    """Create basic graphs.

    This is a demo visualization utility.
    """

    labels = _normalize_label_order(label_map)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig1 = plt.figure(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format="d", cmap="Blues", ax=plt.gca(), colorbar=False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", dpi=150)
    plt.close(fig1)

    # Accuracy chart (single value over samples; demo)
    # We build a prefix accuracy array.
    prefix_acc: List[float] = []
    correct = 0
    for i, (t, p) in enumerate(zip(y_true, y_pred), 1):
        if t == p:
            correct += 1
        prefix_acc.append(correct / i)

    fig2 = plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(prefix_acc) + 1), prefix_acc, marker="o", linewidth=2)
    plt.ylim(0, 1.05)
    plt.xlabel("Sample index")
    plt.ylabel("Prefix accuracy")
    plt.title("Accuracy chart (prefix)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=150)
    plt.close(fig2)

    # Prediction score graph (probability-like confidence over samples)
    fig3 = plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(predicted_scores) + 1), predicted_scores, marker="o", linewidth=2)
    plt.xlabel("Sample index")
    plt.ylabel("Predicted class confidence")
    plt.title("Prediction score graph")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format="png", dpi=150)
    plt.close(fig3)

    return EvalOutputs(
        confusion_matrix_png=buf1.getvalue(),
        accuracy_curve_png=buf2.getvalue(),
        score_curve_png=buf3.getvalue(),
    )

