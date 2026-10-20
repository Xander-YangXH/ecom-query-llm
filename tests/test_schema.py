from ecomq.data.io import read_jsonl
from ecomq.schema import format_output, parse_output, validate_record

SAMPLE = "data_sample/seed_sample.jsonl"


def test_sample_records_are_valid():
    for rec in read_jsonl(SAMPLE):
        assert validate_record(rec) == [], rec["id"]


def _rec(**label_overrides):
    label = {"intent": "商品搜索", "attributes": {"品牌": "耐克"}, "rewrite": "耐克", "relevance": None}
    label.update(label_overrides)
    return {"query": "耐克跑鞋", "label": label}


def test_rejects_bad_intent():
    assert validate_record(_rec(intent="乱写"))


def test_rejects_hallucinated_attribute_value():
    assert validate_record(_rec(attributes={"品牌": "阿迪达斯"}))


def test_rejects_unknown_attribute_key():
    assert validate_record(_rec(attributes={"产地": "耐克"}))


def test_relevance_requires_title():
    assert validate_record(_rec(relevance="精确"))
    rec = _rec(relevance="精确")
    rec["title"] = "耐克 跑鞋"
    assert validate_record(rec) == []


def test_output_roundtrip():
    label = {"intent": "售后", "attributes": {}, "rewrite": "退货", "relevance": None}
    assert parse_output(format_output(label)) == label


def test_parse_output_handles_code_fence_and_garbage():
    label = {"intent": "售后", "attributes": {}, "rewrite": "退货", "relevance": None}
    assert parse_output("```json\n" + format_output(label) + "\n```") == label
    assert parse_output("不是 json") is None
    assert parse_output('{"intent": "售后"}') is None
