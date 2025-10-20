from __future__ import annotations
from typing import Dict, Any, List, Tuple

NUM_DTYPES = {"int", "float"}
CAT_DTYPES = {"bool", "category", "string"}
DATETIME_DTYPES = {"datetime"}

# Helper Functions

def _col_meta(schema: Dict[str, Any], col: str) -> Dict[str, Any]:
    return schema.get("columns", {}).get(col, {})

def _dtype(schema: Dict[str, Any], col: str) -> str: 
    return _col_meta(schema, col).get("dtype", "unknown")

def _unique_frac(schema: Dict[str, Any], col: str) -> float:
    return float(_col_meta(schema, col).get("unique_frac", 1.0))

def _is_numeric(schema: Dict[str, Any], col: str) -> bool:
    return _dtype(schema, col) in NUM_DTYPES

def _is_datetime(schema: Dict[str, Any], col: str) -> bool:
    return _dtype(schema, col) in DATETIME_DTYPES

def _is_categorical(schema: Dict[str, Any], col: str) -> bool:
    dt = _dtype(schema, col)
    if dt in CAT_DTYPES:
        return True
    # heuristic: treat low-cardinality strings as categorical even if typed string
    return dt == "string" and _unique_frac(schema, col) < 0.05

def _group_count_hint(schema: Dict[str, Any], col: str, n_rows: int) -> int:
    # crude estimate: unique_frac * N
    uf = _unique_frac(schema, col)
    return max(1, int(round(uf * max(n_rows, 1))))

# Core Routing

def route_test(
    relation: str,
    variables: List[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Decide an appropriate statistical test from relation + variable types.
    Returns a dict with { suggested_test, reason, notes }.
    """

    n_rows = int(schema.get("rows", 0))
    vars_ok = [v for v in variables if v in schema.get("columns", {})]
    if len(vars_ok) < 2:
        return {
            "suggested_test": "none",
            "reason": "Need at least two valid variables in schema",
            "notes": {},
        }
    x, y = vars_ok[0], vars_ok[1]


    # Map types
    x_num = _is_numeric(schema, x)
    y_num = _is_numeric(schema, y)
    x_cat = _is_categorical(schema, x)
    y_cat = _is_categorical(schema, y)
    
    # Datetime is unsupported directly here
    if _is_datetime(schema, x) or _is_datetime(schema, y):
        return {
            "suggested_test": "unsupported_datetime",
            "reason": "Datetime detected; convert to derived numeric/categorical first",
            "notes": {"x_dtype": _dtype(schema, x), "y_dtype": _dtype(schema, y)},
        }
    
    # --- Relation-specific routing ---
    if relation == "association":
        # numeric-numeric → Spearman (robust monotonic)
        if x_num and y_num:
            return {
                "suggested_test": "spearman_correlation",
                "reason": "Both numeric; monotonic association requested",
                "notes": {"x_dtype": _dtype(schema, x), "y_dtype": _dtype(schema, y)},
            }
        # categorical-numeric → group comparison on numeric by groups
        if (x_cat and y_num) or (x_num and y_cat):
            grp, metric = (x, y) if x_cat and y_num else (y, x)
            k = _group_count_hint(schema, grp, n_rows)
            test = "kruskal_wallis" if k > 2 else "mann_whitney_or_welch_t"
            return {
                "suggested_test": test,
                "reason": f"{grp} is categorical (k≈{k}); {metric} numeric",
                "notes": {"groups_est": k},
            }
        # categorical-categorical → chi-square
        if x_cat and y_cat:
            return {
                "suggested_test": "chi_square",
                "reason": "Both categorical; association via contingency",
                "notes": {},
            }
        return {
            "suggested_test": "unknown_association",
            "reason": "Unrecognized type combo for association",
            "notes": {"x": _dtype(schema, x), "y": _dtype(schema, y)},
        }

    if relation == "group_mean_diff":
        # Expect first var = group, second = numeric metric (but be flexible)
        if x_cat and y_num:
            k = _group_count_hint(schema, x, n_rows)
            test = "anova" if k > 2 else "welch_t_or_mann_whitney"
            return {
                "suggested_test": test,
                "reason": f"group={x} (k≈{k}), metric={y}",
                "notes": {"groups_est": k},
            }
        if y_cat and x_num:
            k = _group_count_hint(schema, y, n_rows)
            test = "anova" if k > 2 else "welch_t_or_mann_whitney"
            return {
                "suggested_test": test,
                "reason": f"group={y} (k≈{k}), metric={x}",
                "notes": {"groups_est": k},
            }
        return {
            "suggested_test": "unknown_group_mean_diff",
            "reason": "Need one categorical (group) and one numeric (metric)",
            "notes": {"x": _dtype(schema, x), "y": _dtype(schema, y)},
        }

    if relation == "proportion_diff":
        # categorical-categorical (outcome is binary-ish preferred) → chi-square or prop test
        if x_cat and y_cat:
            return {
                "suggested_test": "chi_square",
                "reason": "Proportion/rate language with two categorical variables",
                "notes": {},
            }
        return {
            "suggested_test": "unknown_proportion_diff",
            "reason": "Proportion difference expects categorical variables",
            "notes": {"x": _dtype(schema, x), "y": _dtype(schema, y)},
        }

    # Fallback
    return {
        "suggested_test": "unknown",
        "reason": f"Unhandled relation: {relation}",
        "notes": {},
    }

# --- Covariate suggestions -------------------------------------------------

def covariate_suggestion(
    relation: str, variables: List[str], schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Suggest when a regression (OLS/logistic) is more appropriate
    than a simple bivariate test.
    """
    if len(variables) < 2:
        return {"suggestion": "none", "reason": "Need ≥2 variables"}

    x, y = variables[0], variables[1]
    x_num, y_num = _is_numeric(schema, x), _is_numeric(schema, y)
    x_cat, y_cat = _is_categorical(schema, x), _is_categorical(schema, y)

    # If both numeric and you're considering association → OLS with covariates is a natural extension.
    if relation == "association" and x_num and y_num:
        return {"suggestion": "ols", "reason": "Both numeric; model y ~ x + covariates"}

    # If outcome looks categorical and predictor numeric/cat → logistic.
    # Heuristic: treat 'winner'/'expired' keywords as categorical outcomes.
    outcome = y  # assume second variable often plays 'outcome' role
    if relation in {"group_mean_diff", "proportion_diff"} and _is_categorical(schema, outcome):
        return {"suggestion": "logistic", "reason": f"Categorical outcome ({outcome}); consider logit with covariates"}

    return {"suggestion": "none", "reason": "Bivariate test likely sufficient for MVP"}