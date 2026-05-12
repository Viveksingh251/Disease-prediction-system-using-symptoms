from __future__ import annotations

import base64
from typing import Dict

from ml_eval import EvalOutputs


def png_to_data_uri(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def eval_outputs_to_template_context(outputs: EvalOutputs) -> Dict[str, str]:
    return {
        "confusion_matrix_png": png_to_data_uri(outputs.confusion_matrix_png),
        "accuracy_curve_png": png_to_data_uri(outputs.accuracy_curve_png),
        "score_curve_png": png_to_data_uri(outputs.score_curve_png),
    }

