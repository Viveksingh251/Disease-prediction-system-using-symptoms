"""Web interface for the Disease Prediction System.

This wraps the existing CLI logic as a small Flask app.

Run:
  pip install -r requirements.txt
  python interface.py
Then open:
  http://127.0.0.1:5000

Note: This is a demo and not medical advice.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from flask import Flask, redirect, render_template_string, request

from diseases import DISEASE_DB, normalize_symptom
from predictor import predict_disease
from recommenders import build_personalities, recommend_treatment

app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Disease Prediction</title>
  <style>
    :root{--bg:#0b1220;--panel:#0f172a;--muted:#94a3b8;--text:#e5e7eb;--primary:#3b82f6;--primary2:#2563eb;--border:rgba(148,163,184,.25)}
    *{box-sizing:border-box}
    body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial; margin:0; padding:28px; background:linear-gradient(135deg,#0b1220,#111827); color:var(--text); min-height:100vh}
    .wrap{max-width:980px; margin:0 auto}
    .topbar{display:flex; align-items:flex-end; justify-content:space-between; gap:16px; margin-bottom:18px}
    h2{margin:0; font-size:24px; letter-spacing:.2px}
    .sub{color:var(--muted); margin:8px 0 0; line-height:1.5}
    .pill{background:rgba(59,130,246,.15); border:1px solid rgba(59,130,246,.35); color:#bfdbfe; padding:7px 10px; border-radius:999px; font-size:12px; white-space:nowrap}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    label{display:block;font-weight:700;margin-bottom:6px; color:#dbeafe}
    input, select, textarea{width:100%;padding:11px;border:1px solid var(--border); background:rgba(15,23,42,.6); color:var(--text); border-radius:12px; outline:none}
    input:focus, select:focus, textarea:focus{border-color:rgba(59,130,246,.75); box-shadow:0 0 0 3px rgba(59,130,246,.18)}
    textarea{min-height:95px;resize:vertical}
    .card{border:1px solid var(--border); background:rgba(15,23,42,.65); border-radius:16px;padding:16px;margin-top:18px; box-shadow:0 10px 30px rgba(0,0,0,.15)}
    button{padding:10px 14px;border:none;border-radius:12px;background:linear-gradient(135deg,var(--primary),var(--primary2));color:white;cursor:pointer; font-weight:700}
    button:active{transform:translateY(1px)}
    .ghost{background:transparent; border:1px solid var(--border); color:var(--text)}
    .hint{color:var(--muted); font-size:12px;margin-top:6px}
    pre{background:rgba(11,18,32,.9); color:#e5e7eb;padding:12px;border-radius:12px;overflow:auto; border:1px solid rgba(148,163,184,.2)}
    .row{display:flex; gap:12px; align-items:center; flex-wrap:wrap}

  </style>
</head>
<body>
  <h2>Disease Prediction System</h2>
  <p class="hint">Enter symptoms and a basic patient profile. The app predicts a likely disease and shows personalized treatment using a selected personality strategy.</p>

  <form method="post">
    <div class="grid">
      <div>
        <label>Age</label>
        <input name="age" type="number" min="0" value="25" />
        <div class="hint">Used to adjust advice (e.g., older adults/children).</div>
      </div>
      <div>
        <label>Sex</label>
        <select name="sex">
          <option value="male">male</option>
          <option value="female">female</option>
          <option value="other" selected>other</option>
        </select>
      </div>
      <div>
        <label>Risk tolerance</label>
        <select name="risk_tolerance">
          <option value="low">low</option>
          <option value="medium" selected>medium</option>
          <option value="high">high</option>
        </select>
        <div class="hint">Controls which “personality” strategy is used.</div>
      </div>
      <div>
        <label>Symptoms (comma-separated)</label>
        <textarea name="symptoms" placeholder="Try: fever, cough, fatigue"></textarea>
        <div class="hint">Type any symptoms you want—model is a simple baseline demo.</div>
      </div>
    </div>

    <div class="grid" style="margin-top:12px">
      <div>
        <label>Allergies (comma-separated)</label>
        <input name="allergies" type="text" placeholder="e.g., aspirin allergy" />
      </div>
      <div>
        <label>Existing conditions (comma-separated)</label>
        <input name="conditions" type="text" placeholder="e.g., asthma" />
      </div>
    </div>

    <div style="margin-top:14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap">
      <button type="submit">Predict</button>
      <span class="hint">Shows disease + personalized treatment.</span>
      <button type="button" style="background:#111827" onclick="setExample()">Use Example</button>
    </div>

    <script>
      function setExample(){
        document.querySelector('textarea[name="symptoms"]').value = 'fever, cough, body aches';
        document.querySelector('input[name="age"]').value = 28;
        document.querySelector('select[name="risk_tolerance"]').value = 'medium';
        document.querySelector('input[name="allergies"]').value = '';
        document.querySelector('input[name="conditions"]').value = '';
      }
    </script>
  </form>


  {% if result %}
    <div class="card">
      <h3>Prediction</h3>
      <pre>{{ result.prediction }}</pre>
    </div>

    <div class="card">
      <h3>Personalized Treatment</h3>
      <pre>{{ result.treatment }}</pre>
    </div>

    <div class="card">
      <h3>Optional: Other personalities</h3>
      {% for k, v in result.other_treatments.items() %}
        <div style="margin-top:14px">
          <strong>{{ k }}</strong>
          <pre style="margin-top:8px">{{ v }}</pre>
        </div>
      {% endfor %}
    </div>

    <p class="hint">DISCLAIMER: This is a demo and not medical advice. Consult a professional.</p>
  {% endif %}
</body>
</html>
"""


class PatientProfile:
    def __init__(
        self,
        age: int,
        sex: str,
        allergies: List[str],
        conditions: List[str],
        risk_tolerance: str,
    ) -> None:
        self.age = age
        self.sex = sex
        self.allergies = allergies
        self.conditions = conditions
        self.risk_tolerance = risk_tolerance


def parse_list_csv(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template_string(TEMPLATE, result=None)

    age = int((request.form.get("age") or "0").strip() or 0)
    sex = (request.form.get("sex") or "other").strip().lower() or "other"
    risk_tolerance = (request.form.get("risk_tolerance") or "medium").strip().lower() or "medium"

    allergies_raw = parse_list_csv(request.form.get("allergies" or ""))
    conditions_raw = parse_list_csv(request.form.get("conditions" or ""))

    user_symptoms_raw = parse_list_csv(request.form.get("symptoms" or ""))
    user_symptoms = [normalize_symptom(x) for x in user_symptoms_raw if normalize_symptom(x)]

    if not user_symptoms:
        # Keep UI simple: show empty result
        return render_template_string(
            TEMPLATE,
            result={
                "prediction": "No symptoms provided.",
                "treatment": "N/A",
                "other_treatments": {},
            },
        )

    profile = PatientProfile(
        age=age,
        sex=sex,
        allergies=[normalize_symptom(a) for a in allergies_raw],
        conditions=[normalize_symptom(c) for c in conditions_raw],
        risk_tolerance=risk_tolerance,
    )

    pred = predict_disease(user_symptoms, DISEASE_DB)

    personalities = build_personalities()
    personality_key = "conservative" if profile.risk_tolerance == "low" else "balanced" if profile.risk_tolerance == "medium" else "aggressive"
    personality = personalities[personality_key]

    treatment = recommend_treatment(pred.disease, profile, personality)

    other_treatments: Dict[str, str] = {}
    for key, p in personalities.items():
        if key == personality_key:
            continue
        other_treatments[key] = recommend_treatment(pred.disease, profile, p)

    result = {
        "prediction": f"Disease: {pred.disease}\nScore: {pred.score:.3f}\nMatching symptoms: {', '.join(pred.matched_symptoms) or 'None'}",
        "treatment": treatment,
        "other_treatments": other_treatments,
    }

    return render_template_string(TEMPLATE, result=result)


if __name__ == "__main__":
    # Recommended for local dev only
    app.run(host="127.0.0.1", port=5000, debug=True)

