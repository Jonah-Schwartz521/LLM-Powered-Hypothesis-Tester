from hypotest.hypotheses import parse_question

def test_parse_simple():
    q = "Does length_of_stay correlate with procedure_count?"
    res = parse_question(q)
    assert res["relation"] == "association"
    assert "length_of_stay" in res["variables"]
    assert "procedure_count" in res["variables"]
    assert res["suggested_test"] == "spearman_correlation"
