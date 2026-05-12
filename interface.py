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

from flask import Flask, render_template_string, request, session, redirect

from diseases import DISEASE_DB, normalize_symptom
from auth_store import AuthStore
from user_data_store import PredictionLog, UserDataStore
from auth_routes import try_login, try_signup
from ml_eval import evaluate_classifier
from graphs import eval_outputs_to_template_context
from predictor import predict_disease
from recommenders import build_personalities, recommend_treatment

app = Flask(__name__)

# Session secret (demo only; for real deployments set via env var)
app.secret_key = "dev-secret-key-change-me"

auth_store = AuthStore(path="auth_users.json")
user_data_store = UserDataStore(path="user_data.json")

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
  <div class="topbar">
    <h2>Disease Prediction System</h2>
    <div>
      {% if session.get('username') %}
        <span class="pill">Logged in as {{ session.get('username') }}</span>
        <form method="post" action="/logout" style="display:inline;margin-left:10px">
          <button class="ghost" type="submit" style="padding:8px 12px;border-radius:12px">Logout</button>
        </form>
      {% endif %}
    </div>
  </div>

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

    {% if result.graphs %}
      <div class="card">
        <h3>Accuracy & Evaluation Graphs</h3>
        <p class="hint">Generated from the project’s internal synthetic evaluation data.</p>

        <div class="grid" style="margin-top:12px">
          <div>
            <label>Confusion matrix</label>
            <img style="width:100%;border-radius:12px;border:1px solid rgba(148,163,184,.25)" src="{{ result.graphs.confusion_matrix_png }}" alt="confusion matrix"/>
          </div>
          <div>
            <label>Accuracy chart</label>
            <img style="width:100%;border-radius:12px;border:1px solid rgba(148,163,184,.25)" src="{{ result.graphs.accuracy_curve_png }}" alt="accuracy chart"/>
          </div>
        </div>

        <div style="margin-top:12px">
          <label>Prediction score graph</label>
          <img style="width:100%;border-radius:12px;border:1px solid rgba(148,163,184,.25)" src="{{ result.graphs.score_curve_png }}" alt="prediction score graph"/>
        </div>
      </div>
    {% endif %}

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


def _require_login():
    if not session.get("username"):
        return redirect("/login")
    return None


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template_string(
            """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Sign Up - Disease Prediction</title>
  <style>
    :root{
      --bg:#070b18;--panel:rgba(15,23,42,.78);--text:#e5e7eb;
      --muted:#94a3b8;--primary:#7c3aed;--primary2:#3b82f6;
      --border:rgba(148,163,184,.25);--bad:#ef4444;--good:#22c55e;
    }
    *{box-sizing:border-box}
    body{
      margin:0; min-height:100vh; color:var(--text);
      font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;
      background:
        radial-gradient(1200px 650px at 20% 15%, rgba(124,58,237,.35), transparent 60%),
        radial-gradient(900px 520px at 90% 25%, rgba(59,130,246,.35), transparent 55%),
        linear-gradient(135deg,#070b18,#0b1220,#111827);
      display:flex; align-items:center; justify-content:center;
      padding:24px; overflow:hidden;
    }

    /* Background “images” related to the project: pattern grid + blobs */
    .bg-grid{
      position:fixed; inset:0;
      background-image:
        linear-gradient(rgba(148,163,184,.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148,163,184,.08) 1px, transparent 1px);
      background-size:60px 60px;
      opacity:.35;
      pointer-events:none;
      mask-image: radial-gradient(800px 500px at 50% 20%, black 45%, transparent 75%);
    }
    .blob{position:fixed; border-radius:999px; filter:blur(34px); opacity:.55; pointer-events:none; animation: floaty 8s ease-in-out infinite;}
    .b1{width:420px;height:420px;left:-140px;top:-120px;background:rgba(124,58,237,.55)}
    .b2{width:520px;height:520px;right:-220px;top:60px;background:rgba(59,130,246,.45);animation-delay:-2s}
    .b3{width:380px;height:380px;left:10%;bottom:-190px;background:rgba(34,197,94,.22);animation-delay:-4s}
    @keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(18px)}}

    .wrap{width:100%; max-width:620px; position:relative; z-index:2}
    .card{
      background:var(--panel);
      border:1px solid var(--border);
      border-radius:18px;
      box-shadow:0 30px 80px rgba(0,0,0,.45);
      padding:22px;
    }
    .badge{display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(124,58,237,.18);border:1px solid rgba(124,58,237,.35);color:#ddd6fe;font-weight:800;font-size:12px;margin-bottom:10px}
    h1{margin:0; font-size:26px; letter-spacing:.2px}
    .subtitle{margin:10px 0 0; color:var(--muted); line-height:1.55}

    .feature{
      margin:16px 0 18px;
      padding:14px;
      border-radius:16px;
      background:rgba(2,6,23,.25);
      border:1px solid rgba(148,163,184,.18);
    }
    .feature strong{color:#bfdbfe}
    .feature ul{margin:10px 0 0; padding-left:18px; color:var(--muted)}
    .feature li{margin:6px 0}

    label{display:block; font-weight:800; color:#dbeafe; margin:12px 0 6px}
    input{width:100%; padding:12px; border-radius:12px; border:1px solid rgba(148,163,184,.35); background:rgba(2,6,23,.35); color:var(--text); outline:none}
    input:focus{border-color:rgba(59,130,246,.9); box-shadow:0 0 0 3px rgba(59,130,246,.18)}

    .btn{width:100%; margin-top:18px; padding:12px 14px; border:none; border-radius:12px; background:linear-gradient(135deg,var(--primary),var(--primary2)); color:#fff; font-weight:900; cursor:pointer}
    .btn:active{transform:translateY(1px)}

    .err{color:var(--bad); margin-top:10px; font-weight:800}
    .small{color:var(--muted); font-size:12px; margin-top:14px; line-height:1.5}
    a.link{color:#93c5fd; text-decoration:none; font-weight:900}
    a.link:hover{text-decoration:underline}
  </style>
</head>
<body>
  <div class="bg-grid"></div>
  <div class="blob b1"></div>
  <div class="blob b2"></div>
  <div class="blob b3"></div>

  <div class="wrap">
    <div class="card">
      <div class="badge">Create account</div>
      <h1>Disease Prediction System</h1>
      <p class="subtitle">Sign up to use disease predictions, evaluation graphs, and personalized treatment recommendations.</p>

      <div class="feature">
        <strong>What this project supports</strong>
        <ul>
          <li>Login/Signup authentication (stored locally)</li>
          <li>Disease prediction from symptoms</li>
          <li>Personalized treatment by different personalities</li>
          <li>Accuracy & evaluation charts</li>
        </ul>
      </div>

      <form method='post'>
        <label>Username</label>
        <input name='username' placeholder='Choose a username' autofocus/>

        <label>Password</label>
        <input name='password' type='password' placeholder='Create a password'/>

        <button class='btn' type='submit'>Create account</button>
      </form>

      <div class="err">{{ error or '' }}</div>
      <div class="small">Already have an account? <a class="link" href='/login'>Login</a></div>
    </div>
  </div>
</body>
</html>
            """,
            error=None,
        )


    ok, msg = try_signup(request, auth_store)
    if not ok:
        return render_template_string(
            """
            <h2>Sign up</h2>
            <form method='post'>
              <label>Username</label><input name='username' value='{{request.form.get('username','')}}'/><br/>
              <label>Password</label><input name='password' type='password'/><br/>
              <button type='submit'>Create account</button>
            </form>
            <p style='color:#ef4444;'>{{ msg }}</p>
            <p><a href='/login'>Back to login</a></p>
            """,
            msg=msg,
        )

    return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template_string(
            """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Login - Disease Prediction</title>
  <style>
    :root{
      --bg:#070b18;--panel:rgba(15,23,42,.78);--text:#e5e7eb;
      --muted:#94a3b8;--primary:#7c3aed;--primary2:#3b82f6;--border:rgba(148,163,184,.25);
      --good:#22c55e;--bad:#ef4444;
    }
    *{box-sizing:border-box}
    body{
      margin:0; min-height:100vh; color:var(--text);
      font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;
      background:
        radial-gradient(1200px 600px at 10% 10%, rgba(124,58,237,.35), transparent 60%),
        radial-gradient(900px 500px at 90% 20%, rgba(59,130,246,.35), transparent 55%),
        linear-gradient(135deg,#070b18,#0b1220,#111827);
      display:flex; align-items:center; justify-content:center;
      padding:24px;
      overflow:hidden;
    }
    /* “images” in the background (pure CSS blobs + subtle grid) */
    .bg-grid{
      position:fixed; inset:0;
      background-image: linear-gradient(rgba(148,163,184,.08) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(148,163,184,.08) 1px, transparent 1px);
      background-size: 60px 60px;
      opacity:.35;
      pointer-events:none;
      mask-image: radial-gradient(800px 450px at 50% 20%, black 45%, transparent 75%);
    }
    .blob{
      position:fixed; border-radius:999px; filter:blur(34px);
      opacity:.55; pointer-events:none;
      animation: floaty 8s ease-in-out infinite;
    }
    .b1{width:420px;height:420px;left:-140px;top:-120px;background:rgba(124,58,237,.55)}
    .b2{width:520px;height:520px;right:-220px;top:60px;background:rgba(59,130,246,.45);animation-delay:-2s}
    .b3{width:380px;height:380px;left:10%;bottom:-190px;background:rgba(34,197,94,.22);animation-delay:-4s}
    @keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(18px)}}

    .wrap{width:100%; max-width:560px; position:relative; z-index:2}
    .card{
      background:var(--panel); border:1px solid var(--border);
      border-radius:18px; box-shadow:0 30px 80px rgba(0,0,0,.45);
      padding:22px;
    }
    h1{margin:0; font-size:26px; letter-spacing:.2px}
    .subtitle{margin:10px 0 0; color:var(--muted); line-height:1.55}

    .feature-list{margin:14px 0 18px; padding-left:18px; color:var(--text)}
    .feature-list li{margin:6px 0; color:var(--muted)}
    .badge{
      display:inline-block; padding:6px 10px; border-radius:999px;
      background:rgba(124,58,237,.18); border:1px solid rgba(124,58,237,.35);
      color:#ddd6fe; font-weight:700; font-size:12px;
      margin-bottom:10px;
    }

    label{display:block; font-weight:700; color:#dbeafe; margin:12px 0 6px}
    input{
      width:100%; padding:12px 12px; border-radius:12px;
      border:1px solid rgba(148,163,184,.35);
      background:rgba(2,6,23,.35); color:var(--text);
      outline:none;
    }
    input:focus{border-color:rgba(59,130,246,.9); box-shadow:0 0 0 3px rgba(59,130,246,.18)}

    .btn{
      width:100%; margin-top:16px;
      padding:12px 14px; border:none; border-radius:12px;
      background:linear-gradient(135deg,var(--primary),var(--primary2));
      color:#fff; font-weight:800; cursor:pointer;
    }
    .btn:active{transform:translateY(1px)}
    .row{display:flex; gap:12px; align-items:center; margin-top:14px}
    .ghost{
      width:100%; text-align:center; padding:10px 14px;
      border-radius:12px; border:1px solid var(--border);
      background:transparent; color:var(--text); font-weight:800;
      text-decoration:none; display:inline-block;
    }

    .err{color:var(--bad); margin-top:10px; font-weight:700}
    .ok{color:var(--good); margin-top:10px; font-weight:700}

    .small{color:var(--muted); font-size:12px; margin-top:14px; line-height:1.5}
    a.link{color:#93c5fd; text-decoration:none; font-weight:800}
    a.link:hover{text-decoration:underline}
  </style>
</head>
<body>
  <div class="bg-grid"></div>
  <div class="blob b1"></div>
  <div class="blob b2"></div>
  <div class="blob b3"></div>

  <div class="wrap">
    <div class="card">
      <div class="badge">Login to continue</div>
      <h1>Disease Prediction System</h1>
      <p class="subtitle">Upload/enter symptoms and get a disease prediction + personalized treatment.</p>
      <ul class="feature-list">
        <li>Login/Signup authentication (stored locally in <b>auth_users.json</b>)</li>
        <li>Disease prediction from symptoms (ML-backed baseline model)</li>
        <li>Personalized treatment using different “personalities” strategies</li>
        <li>Evaluation charts: confusion matrix, accuracy chart, prediction score graph</li>
      </ul>

      <form method='post'>
        <label>Username</label>
        <input name='username' placeholder='Enter username' autofocus/>
        <label>Password</label>
        <input name='password' type='password' placeholder='Enter password'/>
        <button class='btn' type='submit'>Login</button>
      </form>

      <div class="err">{{ error or '' }}</div>
      <div class="small">
        New here? <a class="link" href='/signup'>Create an account</a>
      </div>
    </div>
  </div>
</body>
</html>
            """,
            error=None,
        )

    ok, msg = try_login(request, auth_store)
    if not ok:
        return render_template_string(
            """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Login - Disease Prediction</title>
  <style>
    :root{--bg:#070b18;--panel:rgba(15,23,42,.78);--text:#e5e7eb;--muted:#94a3b8;--primary:#7c3aed;--primary2:#3b82f6;--border:rgba(148,163,184,.25);--bad:#ef4444}
    *{box-sizing:border-box}
    body{margin:0;min-height:100vh;color:var(--text);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;background:linear-gradient(135deg,#070b18,#0b1220,#111827);display:flex;align-items:center;justify-content:center;padding:24px}
    .card{width:100%;max-width:520px;background:var(--panel);border:1px solid var(--border);border-radius:18px;box-shadow:0 30px 80px rgba(0,0,0,.45);padding:22px}
    .badge{display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(124,58,237,.18);border:1px solid rgba(124,58,237,.35);color:#ddd6fe;font-weight:700;font-size:12px;margin-bottom:10px}
    h1{margin:0;font-size:26px;letter-spacing:.2px}
    .subtitle{margin:10px 0 0;color:var(--muted);line-height:1.55}
    label{display:block;font-weight:700;color:#dbeafe;margin:12px 0 6px}
    input{width:100%;padding:12px;border-radius:12px;border:1px solid rgba(148,163,184,.35);background:rgba(2,6,23,.35);color:var(--text);outline:none}
    input:focus{border-color:rgba(59,130,246,.9);box-shadow:0 0 0 3px rgba(59,130,246,.18)}
    .btn{width:100%;margin-top:16px;padding:12px 14px;border:none;border-radius:12px;background:linear-gradient(135deg,var(--primary),var(--primary2));color:#fff;font-weight:800;cursor:pointer}
    .err{color:var(--bad);margin-top:10px;font-weight:800}
    .small{color:var(--muted);font-size:12px;margin-top:14px;line-height:1.5}
    a{color:#93c5fd;text-decoration:none;font-weight:800}
    a:hover{text-decoration:underline}
  </style>
</head>
<body>
  <div class="card">
    <div class="badge">Login to continue</div>
    <h1>Disease Prediction System</h1>
    <p class="subtitle">Authentication failed. Please try again.</p>

    <form method='post'>
      <label>Username</label>
      <input name='username' value='{{request.form.get("username","")}}' placeholder='Enter username' autofocus/>
      <label>Password</label>
      <input name='password' type='password' placeholder='Enter password'/>
      <button class='btn' type='submit'>Login</button>
    </form>

    <div class="err">{{ msg }}</div>
    <div class="small">New here? <a href='/signup'>Create an account</a></div>
  </div>
</body>
</html>
            """,
            msg=msg,
        )

    session["username"] = (request.form.get("username") or "").strip()
    return redirect("/")


@app.route("/logout", methods=["POST", "GET"])
def logout():
    session.pop("username", None)
    return redirect("/login")


@app.route("/", methods=["GET", "POST"])
def home():
    _redirect = _require_login()
    if _redirect is not None:
        return _redirect

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
    personality_key = (
        "conservative" if profile.risk_tolerance == "low" else "balanced" if profile.risk_tolerance == "medium" else "aggressive"
    )
    personality = personalities[personality_key]

    treatment = recommend_treatment(pred.disease, profile, personality)

    other_treatments: Dict[str, str] = {}
    for key, p in personalities.items():
        if key == personality_key:
            continue
        other_treatments[key] = recommend_treatment(pred.disease, profile, p)

    # Accuracy graphs / evaluation
    graphs_ctx = None
    try:
        from ml_model import build_synthetic_dataset

        X_list, y_keys, vocabulary = build_synthetic_dataset(DISEASE_DB)
        if X_list and y_keys:
            y_true: List[str] = []
            y_pred: List[str] = []
            predicted_scores: List[float] = []

            vocab = list(vocabulary)
            for x_vec, true_key in zip(X_list, y_keys):
                true_symptoms = [vocab[i] for i, v in enumerate(x_vec) if int(v) == 1]
                p = predict_disease(true_symptoms, DISEASE_DB)
                y_true.append(true_key)
                y_pred.append(p.disease)
                predicted_scores.append(float(p.score))

            label_map = sorted(set(y_true) | set(y_pred))

            outputs = evaluate_classifier(
                y_true=y_true,
                y_pred=y_pred,
                predicted_scores=predicted_scores,
                label_map=label_map,
            )
            graphs_ctx = eval_outputs_to_template_context(outputs)
    except Exception:
        graphs_ctx = None

    result = {
        "prediction": f"Disease: {pred.disease}\nScore: {pred.score:.3f}\nMatching symptoms: {', '.join(pred.matched_symptoms) or 'None'}",
        "treatment": treatment,
        "other_treatments": other_treatments,
        "graphs": graphs_ctx,
    }

    # Persist prediction for the logged-in user
    try:
        username = session.get("username")
        if username:
            user_data_store.add_prediction_log(
                PredictionLog(
                    username=str(username),
                    symptoms=user_symptoms,
                    predicted_disease=pred.disease,
                    predicted_score=float(pred.score),
                )
            )
    except Exception:
        pass

    return render_template_string(TEMPLATE, result=result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

