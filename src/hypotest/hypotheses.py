from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional
import typer
from hypotest.router import route_test, covariate_suggestion
from hypotest.runner import run_bivariate
from hypotest.narrator import summarize_result

app = typer.Typer(name="hypothesis", add_completion=False)

def guess_relation(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in ["correlat", "association", "related"]):
        return "association"
    if any(k in ql for k in ["difference", "higher than", "lower than", "greater than", "less than", "compare", "diff"]):
        return "group_mean_diff"
    if any(k in ql for k in ["proportion", "rate", "odds", "likelihood"]):
        return "proportion_diff"
    return "unknown"

def extract_variables_simple(q: str):
    ql = q.lower()
    candidates = ["length_of_stay", "procedure_count", "total_charges",
                  "admission_day", "discharge_disposition", "weight_class",
                  "winner", "method", "rounds", "sig_str_red", "sig_str_blue"]
    return [c for c in candidates if c.replace("_"," ") in ql or c in ql]

def build_hypotheses(relation: str, vars):
    if relation == "association" and len(vars) >= 2:
        x, y = vars[0], vars[1]
        return {
            "H0": f"There is no monotonic association between {x} and {y} (ρ = 0).",
            "H1": f"There is a monotonic association between {x} and {y} (ρ ≠ 0)."
        }
    if relation == "group_mean_diff" and len(vars) >= 2:
        grp, metric = vars[0], vars[1]
        return {
            "H0": f"The mean of {metric} is equal across groups of {grp}.",
            "H1": f"The mean of {metric} differs for at least one group of {grp}."
        }
    if relation == "proportion_diff" and len(vars) >= 2:
        grp, outcome = vars[0], vars[1]
        return {
            "H0": f"The proportion of {outcome} is equal across groups of {grp}.",
            "H1": f"The proportion of {outcome} differs for at least one group of {grp}."
        }
    return {"H0":"Unable to form hypothesis", "H1":"Unable to form hypothesis"}

def suggest_test(relation: str, vars):
    if relation == "association":
        return "spearman_correlation"
    if relation == "group_mean_diff":
        return "kruskal_wallis_or_anova"
    if relation == "proportion_diff":
        return "chi_square"
    return "rule_based_unknown"

def parse_question(question: str, schema: Optional[Dict[str, Any]] = None):
    relation = guess_relation(question)
    vars_found = extract_variables_simple(question)

    # If a schema is provided, keep only variables that exist in the schema
    if schema and "columns" in schema:
        cols = set(schema["columns"].keys())
        vars_found = [v for v in vars_found if v in cols]

    hyp = build_hypotheses(relation, vars_found)
    test = suggest_test(relation, vars_found)
    return {
        "question": question,
        "relation": relation,
        "variables": vars_found,
        "hypotheses": hyp,
        "suggested_test": test,
        "alpha": 0.05,
        "sided": "two-sided",
    }

@app.command("parse")
def cli_parse(
    q: str = typer.Option(..., "--q", help="Natural language hypothesis"),
    schema_path: str | None = typer.Option(None, "--schema", help="Path to schema JSON (optional)"),
    out: str | None = typer.Option(None, "--out", help="Write JSON here (optional)"),
    data_path: str | None = typer.Option(None, "--data", help="Path to CSV to run the suggested test (optional)"),
):
    # 1) Load schema if provided
    schema: Optional[Dict[str, Any]] = None
    if schema_path:
        p = Path(schema_path)
        if p.exists():
            try:
                schema = json.loads(p.read_text())
            except Exception as e:
                raise typer.BadParameter(f"Failed to read schema JSON at {schema_path}: {e}")

    # 2) Build the parse result (schema-aware)
    result = parse_question(q, schema)

    # 2b) Routing + covariate suggestion
    schema_safe = schema or {"columns": {}, "rows": 0}
    routed = route_test(result["relation"], result["variables"], schema_safe)
    covar = covariate_suggestion(result["relation"], result["variables"], schema_safe)
    result["routed_test"] = routed
    result["covariate_suggestion"] = covar

    # 2c) Optional: execute the suggested test end-to-end on a CSV
    if data_path:
        if schema is None:
            raise typer.BadParameter("--data requires --schema so we know column types for routing")
        p_data = Path(data_path)
        if not p_data.exists():
            raise typer.BadParameter(f"Data file not found: {data_path}")
        import pandas as pd
        df = pd.read_csv(p_data)
        exec_res = run_bivariate(df, result["relation"], result["variables"], schema)
        result["execution"] = exec_res
        result["summary_text"] = summarize_result(exec_res)

    # 3) Output
    text = json.dumps(result, indent=2)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)
        typer.echo(f"Wrote {out}")
    else:
        typer.echo(text)
    

if __name__ == "__main__":
    app()
