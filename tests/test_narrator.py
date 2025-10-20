from hypotest.narrator import summarize_result

def test_summary_has_key_phrases():
    res = {"test":"welch_t","p_value":0.03,"stat":2.1,"effect":0.5,"reason":"group_mean_diff","notes":{}}
    txt = summarize_result(res)
    assert "welch t" in txt and ("statistically significant" in txt or "not statistically significant" in txt)