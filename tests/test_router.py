import json
from hypotest.router import route_test, covariate_suggestion

# minimal fake schema for tests
SCHEMA = {
    "rows": 1000,
    "columns": {
        "length_of_stay": {"dtype": "int", "unique_frac": 0.15},
        "procedure_count": {"dtype": "int", "unique_frac": 0.05},
        "total_charges": {"dtype": "float", "unique_frac": 0.9},
        "discharge_disposition": {"dtype": "category", "unique_frac": 0.02},
        "admission_type": {"dtype": "category", "unique_frac": 0.03},
        "winner": {"dtype": "category", "unique_frac": 0.02},
        "weight_class": {"dtype": "category", "unique_frac": 0.08},
        "encounter_date": {"dtype": "datetime", "unique_frac": 0.9},
    }
}

def test_num_num_association_spearman():
    res = route_test("association", ["length_of_stay", "procedure_count"], SCHEMA)
    assert res["suggested_test"] == "spearman_correlation"

def test_cat_num_two_groups_t_or_mw():
    # pretend small groups (unique_frac ~ 0.02 → ~20 groups; we still allow kw if >2)
    res = route_test("group_mean_diff", ["discharge_disposition", "total_charges"], SCHEMA)
    assert res["suggested_test"] in {"anova", "welch_t_or_mann_whitney", "kruskal_wallis"}

def test_num_cat_group_mean_diff():
    res = route_test("group_mean_diff", ["total_charges", "discharge_disposition"], SCHEMA)
    assert res["suggested_test"] in {"anova", "welch_t_or_mann_whitney", "kruskal_wallis"}

def test_cat_cat_association_chi2():
    res = route_test("association", ["discharge_disposition", "admission_type"], SCHEMA)
    assert res["suggested_test"] == "chi_square"

def test_proportion_diff_cat_cat_chi2():
    res = route_test("proportion_diff", ["winner", "weight_class"], SCHEMA)
    assert res["suggested_test"] == "chi_square"

def test_unknown_with_datetime():
    res = route_test("association", ["encounter_date", "total_charges"], SCHEMA)
    assert res["suggested_test"] == "unsupported_datetime"

def test_need_two_vars():
    res = route_test("association", ["length_of_stay"], SCHEMA)
    assert res["suggested_test"] == "none"

def test_unknown_relation():
    res = route_test("weird_relation", ["length_of_stay", "procedure_count"], SCHEMA)
    assert res["suggested_test"] == "unknown"

def test_covariate_suggest_ols():
    res = covariate_suggestion("association", ["total_charges", "length_of_stay"], SCHEMA)
    assert res["suggestion"] in {"ols", "none"}  # allow none if you change heuristic

def test_covariate_suggest_logistic():
    res = covariate_suggestion("proportion_diff", ["weight_class", "winner"], SCHEMA)
    assert res["suggestion"] in {"logistic", "none"}

def test_association_cat_num_routes_group_test():
    res = route_test("association", ["weight_class", "total_charges"], SCHEMA)
    assert res["suggested_test"] in {"mann_whitney_or_welch_t", "kruskal_wallis"}

def test_association_num_cat_routes_group_test():
    res = route_test("association", ["total_charges", "weight_class"], SCHEMA)
    assert res["suggested_test"] in {"mann_whitney_or_welch_t", "kruskal_wallis"}