# LLM-Powered Hypothesis Tester (M0)

**Milestone 0 (M0) goals:**
1) Create repo scaffold and README stub  
2) Add `data/` folder with UFC & SPARCS sample CSVs  
3) Implement `load_data()` and `infer_schema()` that write a `schema.json`

## Quick Start
```bash
# 1) Create a virtual env (recommended)
python3 -m venv .venv && source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Generate schema from sample data
python -m hypotest.io --data data/samples/ufc_sample.csv --out artifacts/schema/ufc_schema.json
python -m hypotest.io --data data/samples/sparcs_sample.csv --out artifacts/schema/sparcs_schema.json
```
