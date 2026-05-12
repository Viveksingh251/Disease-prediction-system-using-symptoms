from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PredictionLog:
    username: str
    symptoms: List[str]
    predicted_disease: str
    predicted_score: float


class UserDataStore:
    """Local JSON persistence for user prediction history.

    Schema (file):
      {
        "prediction_logs": [ { ...PredictionLog... } ]
      }

    This is intentionally simple for the demo.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {"prediction_logs": []}
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: Dict[str, Any]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    def add_prediction_log(self, log: PredictionLog) -> None:
        with self._lock:
            data = self._load()
            data.setdefault("prediction_logs", [])
            data["prediction_logs"].append(asdict(log))
            self._save(data)

    def list_prediction_logs(self, username: Optional[str] = None) -> List[PredictionLog]:
        with self._lock:
            data = self._load()
            logs = data.get("prediction_logs", [])

        out: List[PredictionLog] = []
        for item in logs:
            try:
                pl = PredictionLog(
                    username=str(item.get("username", "")),
                    symptoms=list(item.get("symptoms", [])),
                    predicted_disease=str(item.get("predicted_disease", "")),
                    predicted_score=float(item.get("predicted_score", 0.0)),
                )
                if username is None or pl.username == username:
                    out.append(pl)
            except Exception:
                continue
        return out

