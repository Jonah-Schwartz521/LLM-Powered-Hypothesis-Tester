from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional
import typer

app = typer.Typer(add_completion=False)

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
    out: Optional[str] = typer.Option(None, "--out", help="Path to write JSON output"),
):
    result = parse_question(q)
    text = json.dumps(result, indent=2)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)
        typer.echo(f"Wrote {out}")
    else:
        typer.echo(text)

if __name__ == "__main__":
    app()
