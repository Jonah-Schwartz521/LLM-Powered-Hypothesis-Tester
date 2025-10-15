import pandas as pd
from hypotest.io import load_data, infer_schema

def test_infer_schema_runs(tmp_path):
    csv = tmp_path / "demo.csv"
    pd.DataFrame({"a":[1,2,3], "b":["x","y","y"]}).to_csv(csv, index=False)
    df = load_data(csv)
    schema = infer_schema(df)
    assert "columns" in schema and "a" in schema["columns"]
