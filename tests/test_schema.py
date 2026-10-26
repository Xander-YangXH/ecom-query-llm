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


def test_redact_hides_machine_paths():
    from ecomq.paths import DATA_ROOT, redact

    assert redact(DATA_ROOT / "e1_relevance" / "splits" / "train.jsonl") == "$DATA_ROOT/e1_relevance/splits/train.jsonl"
    assert redact("/private/tmp/xyz/abc/e4-20k-raw.jsonl") == "e4-20k-raw.jsonl"
    assert redact("/home/someone/other/file.jsonl") == "file.jsonl"


def test_relevance_instruction_is_frozen():
    """已训练的所有模型都用这段指令；任何改动都会造成训练/推理不一致（DEVLOG P1），必须原样保留。"""
    from ecomq.schema import INSTRUCTIONS

    assert INSTRUCTIONS["relevance"] == (
        "You are a relevance judge for e-commerce search. Given a user query and a product title, "
        "classify how well the product matches the query. "
        "exact: the product satisfies all requirements of the query. "
        "substitute: the product misses some requirement but could serve as a functional substitute. "
        "complement: the product does not match the query but could be used together with a matching product. "
        "irrelevant: the product is unrelated or fails a central requirement of the query. "
        'Output JSON only, in the form {"relevance": "<label>"}.'
    )


def test_relevance_cot_roundtrip_and_validation():
    cot = {"reason": "A camera battery is an accessory used with the camera the query asks for.",
           "relevance": "complement"}
    text = format_output(cot, "relevance_cot")
    assert text.index('"reason"') < text.index('"relevance"')     # 理由在前，标签以理由为条件
    assert parse_output(text, "relevance_cot") == cot
    assert parse_output(text, "relevance") == cot                 # 标签解析与 relevance 任务兼容
    assert parse_output('{"relevance": "exact"}', "relevance_cot") is None
    assert parse_output('{"reason": 3, "relevance": "exact"}', "relevance_cot") is None
    rec = {"task": "relevance_cot", "query": "camera", "title": "Battery for X100", "label": cot}
    assert validate_record(rec) == []
    assert validate_record({**rec, "label": {**cot, "reason": " "}})
    assert validate_record({**rec, "label": {**cot, "reason": "word " * 41}})
    assert validate_record({**rec, "label": {**cot, "relevance": "partial"}})
