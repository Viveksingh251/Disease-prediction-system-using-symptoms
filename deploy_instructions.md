# Deployment / Packaging Notes

This project is a **demo CLI** (runs locally from Python).

## Requirements
- Python >= 3.10
- No external dependencies (see `requirements.txt`).

## How to run locally
```bat
cd "c:/Users/Vivek Singh/OneDrive/Disease prediction system using symptoms"
python app.py
```

## Recommended packaging (optional)
If you want to turn it into a distributable CLI, the simplest approach is:
- Use `pyproject.toml`
- Create a zip of the folder
- Or build an executable using PyInstaller (optional)

### Optional: install as a package (editable)
```bat
pip install -e .
```

### Optional: PyInstaller (not included)
```bat
pip install pyinstaller
pyinstaller --onefile app.py
```

## Disclaimer
Medical content in this project is illustrative only and not medical advice.

