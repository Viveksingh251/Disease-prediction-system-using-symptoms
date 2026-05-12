# Disease Prediction System (Python)

End-to-end demo project that predicts likely diseases from symptoms and provides:
- Personalized treatment recommendation (rules-based)
- Plug-in architecture for adding more recommendation "personalities"

## Features
- CLI interface to enter symptoms and patient preferences
- Simple ML-free baseline predictor (Jaccard similarity on symptom sets)
- Treatment recommender with disease-specific templates
- Extensible strategy/personalities system

## Run
### 1) (Optional) Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Run the app
```bash
python app.py
```

## Deployment
See `deploy_instructions.md`.

## Notes
This is a self-contained starter project.

