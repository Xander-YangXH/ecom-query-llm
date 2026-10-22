from ecomq.analysis.es_audit import score, select


def _rec(i, gold, group):
    return {"id": str(i), "group": group, "query": "q", "title": "t", "label": {"relevance": gold},
            "meta": {"query_id": i, "product_id": f"P{i}"}}


def test_select_groups_and_one_per_query():
    recs = [_rec(i, "exact", f"g{i % 3}") for i in range(6)] + [_rec(10 + i, "substitute", f"h{i}") for i in range(4)]
    votes = {r["id"]: ["substitute"] * 5 + ["exact"] if r["label"]["relevance"] == "exact" else ["substitute"] * 6 for r in recs}
    items, sizes = select(recs, votes, k=5, n_err=10, n_ctrl=10, seed=0)
    assert sizes["E2S"] == 6 and sizes["control_S"] == 4
    e2s = [r for name, r in items if name == "E2S"]
    assert len(e2s) == 3                      # 3 个不同的 query，每个 query 只取 1 条


def test_score_excess_disagreement():
    key = {"items": {"1": {"set": "E2S", "gold": "E", "model_majority": "S"},
                     "2": {"set": "S2E", "gold": "S", "model_majority": "E"},
                     "3": {"set": "control_E", "gold": "E", "model_majority": "E"},
                     "4": {"set": "control_S", "gold": "S", "model_majority": "S"}}}
    out = score(key, {"1": "S", "2": "S", "3": "E", "4": "U"})
    s = out["summary"]
    assert s["error_disagree_with_gold"] == 0.5 and s["control_disagree_with_gold"] == 0.5
    assert s["excess_disagreement"] == 0.0 and s["error_agree_with_model"] == 0.5
    assert out["control_S"]["uncertain"] == 1
