from collections import Counter

from ecomq.data.make_rationales import clean_reason, reason_flags, review_table, stratified_sample, to_cot_record
from ecomq.schema import validate_record


def _rec(i, lab):
    return {"id": f"esci-{i}", "task": "relevance", "query": f"q{i}", "title": f"t{i}",
            "label": {"relevance": lab}, "group": f"q{i}", "meta": {"query_id": i}}


def test_stratified_sample_balances_classes_and_is_deterministic():
    recs = [_rec(i, lab) for i, lab in enumerate(["exact"] * 300 + ["substitute"] * 80 + ["complement"] * 20 + ["irrelevant"] * 60)]
    a = stratified_sample(recs, 100, seed=0)
    assert Counter(r["label"]["relevance"] for r in a) == {"exact": 25, "substitute": 25, "complement": 20, "irrelevant": 25}
    assert [r["id"] for r in a] == [r["id"] for r in stratified_sample(recs, 100, seed=0)]
    assert stratified_sample(recs, 0, seed=0) is recs


def test_reason_flags_catch_label_words_and_length():
    assert reason_flags("A camera battery used with the camera the query asks for.") == []
    assert reason_flags("This is an exact match for the query.") == ["label_word"]
    assert reason_flags("It could substitute for the requested phone.") == ["label_word"]
    assert reason_flags("Complementary accessory.") == ["label_word"]
    assert "too_long" in reason_flags("word " * 41)
    assert reason_flags("") == ["empty"]
    assert clean_reason('  "Two   spaces\nand a quote."  ') == "Two spaces and a quote."


def test_cot_record_is_valid_and_keeps_original():
    rec = _rec(1, "complement")
    out = to_cot_record(rec, "A charger for the phone the query asks for.", [], retried=False)
    assert out["task"] == "relevance_cot" and validate_record(out) == []
    assert out["label"] == {"reason": "A charger for the phone the query asks for.", "relevance": "complement"}
    assert rec["task"] == "relevance" and "rationale" not in rec["meta"]   # 不修改输入
    table = review_table([out], per_class=5, seed=0)
    assert "complement" in table and table.count("\n") == 3


def test_mask_label_words_and_prompt_render():
    from ecomq.data.make_rationales import RATIONALE_PROMPT, RETRY_SUFFIX, mask_label_words

    assert mask_label_words("It complements the phone, exactly as asked.") == "It [...] the phone, [...] as asked."
    text = RATIONALE_PROMPT.format(query="q", title="t", label="complement")
    assert "human label: complement" in text and "{" not in text
    assert RETRY_SUFFIX.format(words="complements").startswith("\nYour previous answer")


def test_v3_prompt_has_no_template_examples():
    from ecomq.data.make_rationales import RATIONALE_PROMPT, TEMPLATE_PHRASES

    assert not TEMPLATE_PHRASES.search(RATIONALE_PROMPT)   # v2 的例子会被照抄成模板（DEVLOG S22）
    assert TEMPLATE_PHRASES.search("It matches every requirement.") and TEMPLATE_PHRASES.search("unrelated to it")
