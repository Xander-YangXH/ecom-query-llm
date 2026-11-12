from ecomq.data.build_e9 import select
from ecomq.eval.score_labels_cot import common_prefix_len, extract_reason


def test_extract_reason_handles_valid_partial_and_garbage():
    assert extract_reason('{"reason": "A case for the phone.", "relevance": "complement"}') == \
        ("A case for the phone.", "complement", "ok")
    assert extract_reason('{"reason": "A case for the phone.", "relevance": "partial"}') == ("A case for the phone.", None, "ok")
    assert extract_reason('{"reason": "Says \\"hi\\" twice", "relev') == ('Says "hi" twice', None, "reason_only")
    assert extract_reason("no json at all") == ("", None, "parse_failed")
    # 47258 失败的原因：截断的输出里有 JSON 不允许的转义 \'（P17）
    text = r"""{"reason": "It\'s a case for the phone", "relevance": "compl"""
    assert extract_reason(text) == ("It's a case for the phone", None, "reason_only")


def test_common_prefix_len():
    assert common_prefix_len([[1, 2, 3, 9], [1, 2, 4], [1, 2, 3]]) == 2
    assert common_prefix_len([[5, 6], [5, 6, 7]]) == 2


def _row(i, lab, flags=(), blind=None, source="hinted"):
    meta = {"flags": list(flags), "source": source}
    if blind:
        meta["blind"] = {"with_reason": blind}
    return {"id": f"esci-{i}", "task": "relevance_cot", "query": "q", "title": "t",
            "label": {"reason": "A short reason.", "relevance": lab}, "meta": {"rationale": meta}}


def test_select_drops_flagged_and_marks_inconsistent():
    rows = [_row(1, "exact", blind="exact", source="self"), _row(2, "complement", blind="substitute"),
            _row(3, "complement", flags=["label_word"], blind="complement"), _row(4, "irrelevant")]
    kept, st = select(rows)
    assert [r["id"] for r in kept] == ["esci-1", "esci-2", "esci-4"]
    assert [r["meta"]["rationale"]["inconsistent"] for r in kept] == [False, True, False]
    assert st["dropped_flagged"] == {"complement": 1}
    assert st["inconsistent_by_label"]["complement"] == 1 and st["self_by_label"]["exact"] == 1
