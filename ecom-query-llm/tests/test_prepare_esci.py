import pandas as pd

from ecomq.data.prepare_esci import sample_queries, to_record
from ecomq.schema import validate_record


def _df():
    rows = []
    for q in range(20):
        for k in range(6):
            rows.append({"example_id": q * 10 + k, "query": f"query {q}", "query_id": q, "product_id": f"P{q}-{k}",
                         "esci_label": "ESCI"[k % 4], "split": "train", "product_title": f"title {q} {k}"})
    return pd.DataFrame(rows)


def test_sample_queries_respects_limits_and_exclusion():
    df = _df()
    val, val_q = sample_queries(df, 5, 4, seed=0)
    train, train_q = sample_queries(df, 10, 4, seed=1, exclude=val_q)
    assert len(val_q) == 5 and len(train_q) == 10
    assert not (val_q & train_q)
    assert val.groupby("query_id").size().max() == 4
    assert set(train["query_id"]).isdisjoint(set(val["query_id"]))


def test_sample_queries_is_deterministic():
    a, _ = sample_queries(_df(), 5, 4, seed=3)
    b, _ = sample_queries(_df(), 5, 4, seed=3)
    assert list(a["example_id"]) == list(b["example_id"])


def test_to_record_is_valid_and_truncates_title():
    row = _df().iloc[1].to_dict()
    row["product_title"] = "word " * 100
    rec = to_record(row)
    assert validate_record(rec) == []
    assert rec["task"] == "relevance" and rec["label"] == {"relevance": "substitute"}
    assert len(rec["title"]) <= 200


def test_rebalance_hits_targets_and_keeps_size():
    from ecomq.data.prepare_esci import parse_targets, rebalance

    df = _df()
    train, train_q = sample_queries(df, 10, 4, seed=1)
    pool = df[~df["example_id"].isin(train["example_id"])]
    targets = parse_targets("complement=0.25,irrelevant=0.25")
    assert targets == {"C": 0.25, "I": 0.25}
    out = rebalance(train, df, targets, seed=0)
    assert len(out) == len(train)
    shares = out["esci_label"].value_counts(normalize=True)
    assert abs(shares["C"] - 0.25) < 0.03 and abs(shares["I"] - 0.25) < 0.03
    assert out["example_id"].is_unique
    added = set(out["example_id"]) - set(train["example_id"])
    assert added <= set(pool["example_id"])  # 新增样本都是此前未使用的真实样本


def test_grow_train_is_nested_and_respects_exclusions():
    from ecomq.data.prepare_esci import grow_train

    df = _df()
    base_df, base_q = sample_queries(df, 5, 4, seed=1)
    base = [to_record(r) for r in base_df.to_dict("records")]
    excluded = {0, 1, 2} - base_q
    grown, grown_q = grow_train(base, df, 12, 4, seed=7, exclude=excluded)
    assert len(grown_q) == 12 and base_q <= grown_q
    assert grown[: len(base)] == base               # 小集合原样保留在前面
    assert not (grown_q & excluded)
    assert len({r["id"] for r in grown}) == len(grown)
