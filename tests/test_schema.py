from ecomq.data.io import read_jsonl
from ecomq.schema import ESCI_TO_RELEVANCE, RELEVANCE, format_output, parse_output, validate_record

SAMPLE = "data_sample/seed_sample.jsonl"


def test_sample_records_are_valid():
    for rec in read_jsonl(SAMPLE):
        assert validate_record(rec) == [], rec["id"]


def _rec(**label_overrides):
    label = {"intent": "product_search", "attributes": {"brand": "nike"}, "rewrite": "nike", "relevance": None}
    label.update(label_overrides)
    return {"task": "full", "query": "nike running shoes", "label": label}


def test_rejects_bad_intent():
    assert validate_record(_rec(intent="buy_now"))


def test_rejects_hallucinated_attribute_value():
    assert validate_record(_rec(attributes={"brand": "adidas"}))


def test_attribute_match_is_case_insensitive():
    assert validate_record(_rec(attributes={"brand": "Nike"})) == []


def test_rejects_unknown_attribute_key():
    assert validate_record(_rec(attributes={"origin": "nike"}))


def test_relevance_requires_title_in_full_task():
    assert validate_record(_rec(relevance="exact"))
    rec = _rec(relevance="exact")
    rec["title"] = "Nike Running Shoes"
    assert validate_record(rec) == []


def test_relevance_task_validation():
    ok = {"task": "relevance", "query": "hdmi cable", "title": "HDMI Cable 6ft", "label": {"relevance": "substitute"}}
    assert validate_record(ok) == []
    assert validate_record({**ok, "label": {"relevance": "partial"}})
    assert validate_record({**ok, "title": None})
    assert validate_record({**ok, "task": "unknown"})


def test_esci_mapping_covers_label_set():
    assert sorted(ESCI_TO_RELEVANCE.values()) == sorted(RELEVANCE)


def test_output_roundtrip_per_task():
    full = {"intent": "after_sales", "attributes": {}, "rewrite": "return", "relevance": None}
    assert parse_output(format_output(full, "full"), "full") == full
    rel = {"relevance": "complement"}
    assert format_output(rel, "relevance") == '{"relevance": "complement"}'
    assert parse_output(format_output(rel, "relevance"), "relevance") == rel


def test_parse_output_handles_code_fence_and_garbage():
    rel = {"relevance": "exact"}
    assert parse_output("```json\n" + format_output(rel, "relevance") + "\n```", "relevance") == rel
    assert parse_output("not json", "relevance") is None
    assert parse_output('{"intent": "inquiry"}', "full") is None
    assert parse_output('{"label": "exact"}', "relevance") is None


def test_product_fields_render_only_when_present():
    from ecomq.schema import format_input

    base = format_input("q", "t")
    assert format_input("q", "t", None) == base
    assert format_input("q", "t", {}) == base          # 旧数据（没有 product）渲染结果不变
    rich = format_input("q", "t", {"brand": "Sony", "bullets": "a | b"})
    assert rich == base + "\nbrand: Sony\nproduct details: a | b"
    assert "color" not in rich                          # 缺失字段不出现，不写 unknown
