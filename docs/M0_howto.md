# Milestone 0 — Action Steps

## 1) Create and activate a virtual environment
- VS Code Terminal → `python3 -m venv .venv && source .venv/bin/activate`
- If on Windows PowerShell: `python -m venv .venv; .venv\Scripts\Activate.ps1`

## 2) Install dependencies
`pip install -r requirements.txt`

## 3) Generate schemas from the sample CSVs
```
python -m hypotest.io infer --data data/samples/ufc_sample.csv --out artifacts/schema/ufc_schema.json
python -m hypotest.io infer --data data/samples/sparcs_sample.csv --out artifacts/schema/sparcs_schema.json
```

## 4) Validate a fresh CSV against a saved schema (optional, coming M1)
- Load your real CSV with `python -m hypotest.io infer --data <your.csv> --out artifacts/schema/tmp.json`
- Compare fields to flag obvious issues (see `validate_against_schema` in `src/hypotest/io.py`).

## 5) Commit
- Initialize git, make first commit: scaffold + sample data + schema JSONs.
