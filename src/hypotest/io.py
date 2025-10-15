from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import typer

app = typer.Typer(add_completion=False)

# --- Public API ---

def load_data(path: str | Path) -> pd.DataFrame:
    """
    Load CSV into a DataFrame with basic dtype hints.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    return df


def infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Infer a simple schema: column -> {dtype, non_null, unique_frac}
    Dtypes are simplified to: 'int', 'float', 'bool', 'category', 'datetime', 'string'
    """
    schema: Dict[str, Any] = {"columns": {}}

    def simplify_dtype(s: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(s):
            return "bool"
        if pd.api.types.is_integer_dtype(s):
            return "int"
        if pd.api.types.is_float_dtype(s):
            return "float"
        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"
        # heuristic: low-cardinality strings => category candidate
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            unique = s.dropna().nunique()
            if unique <= max(20, int(len(s) * 0.05)):
                return "category"
            return "string"
        return "string"

    for col in df.columns:
        s = df[col]
        dtype = simplify_dtype(s)
        non_null = int(s.notna().sum())
        unique_frac = float(s.nunique(dropna=True) / max(len(s), 1))
        schema["columns"][col] = {
            "dtype": dtype,
            "non_null": non_null,
            "unique_frac": round(unique_frac, 4),
        }

    schema["rows"] = int(len(df))
    return schema


def save_schema(schema: Dict[str, Any], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(schema, f, indent=2)


def validate_against_schema(df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very lightweight validator: check columns present and non-null > 0 for each schema column.
    """
    issues = []
    for col, meta in schema.get("columns", {}).items():
        if col not in df.columns:
            issues.append({"column": col, "issue": "missing_column"})
            continue
        if df[col].notna().sum() == 0:
            issues.append({"column": col, "issue": "all_null_values"})
    return {"ok": len(issues) == 0, "issues": issues}


# --- CLI ---

@app.callback()
def main() -> None:
    """Schema tools for M0: load → infer → save."""


@app.command("infer")
def cli_infer(
    data: str = typer.Option(..., "--data", help="Path to CSV file"),
    out: str = typer.Option(..., "--out", help="Path to write schema JSON"),
) -> None:
    df = load_data(data)
    schema = infer_schema(df)
    save_schema(schema, out)
    typer.echo(f"Wrote schema to {out}")


if __name__ == "__main__":
    app()
