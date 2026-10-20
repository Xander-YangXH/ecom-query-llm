import json

from ecomq.data.io import read_jsonl
from ecomq.data.split import assign
from ecomq.data.to_alpaca import convert
from ecomq.data.validate import run
from ecomq.eval.metrics import evaluate_full, evaluate_records, evaluate_relevance
from ecomq.schema import format_output, task_of

SAMPLE = "data_sample/seed_sample.jsonl"


def test_validate_dedups_and_rejects():
    rows = list(read_jsonl(SAMPLE))
    dup = dict(rows[0], id="dup")
    bad = dict(rows[1], id="bad", label={**rows[1]["label"], "intent": "buy_now"})
    kept, rejects = run(rows + [dup, bad])
    assert len(kept) == len(rows)
    assert {r["id"] for r in rejects} == {"dup", "bad"}


def test_split_keeps_group_together():
    rows = list(read_jsonl(SAMPLE))
    ratios = (0.5, 0.25, 0.25)
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], set()).add(assign(r, ratios, seed=1))
    assert all(len(s) == 1 for s in by_group.values())


def test_split_is_deterministic():
    r = next(iter(read_jsonl(SAMPLE)))
    assert assign(r, (0.9, 0.05, 0.05), 7) == assign(r, (0.9, 0.05, 0.05), 7)


def test_to_alpaca_uses_task_specific_instruction_and_output():
    rows = list(read_jsonl(SAMPLE))
    full = convert(next(r for r in rows if task_of(r) == "full"))
    rel = convert(next(r for r in rows if task_of(r) == "relevance"))
    assert set(json.loads(full["output"])) == {"intent", "attributes", "rewrite", "relevance"}
    assert set(json.loads(rel["output"])) == {"relevance"}
    assert full["instruction"] != rel["instruction"]
    assert rel["input"].startswith("query: ") and "\nproduct title: " in rel["input"]


def test_full_metrics_perfect_and_broken():
    golds = [r["label"] for r in read_jsonl(SAMPLE) if task_of(r) == "full"]
    perfect = evaluate_full(golds, [format_output(g, "full") for g in golds])
    assert perfect["format_valid_rate"] == 1.0
    assert perfect["intent_acc"] == 1.0
    assert perfect["attr_f1"] == 1.0
    assert perfect["relevance_acc"] == 1.0
    broken = evaluate_full(golds, ["garbage"] * len(golds))
    assert broken["format_valid_rate"] == 0.0
    assert broken["intent_acc"] == 0.0
    assert broken["attr_f1"] == 0.0


def test_relevance_metrics():
    golds = [{"relevance": x} for x in ["exact", "exact", "substitute", "irrelevant"]]
    outs = ['{"relevance": "exact"}', '{"relevance": "substitute"}', '{"relevance": "substitute"}', "oops"]
    m = evaluate_relevance(golds, outs)
    assert m["accuracy"] == 0.5
    assert m["format_valid_rate"] == 0.75
    assert m["per_class"]["exact"]["recall"] == 0.5
    assert m["per_class"]["substitute"]["precision"] == 0.5
    assert m["per_class"]["complement"]["support"] == 0
    assert m["confusion"]["irrelevant"] == {"<invalid>": 1}
    # macro-F1 固定在 4 个类别上平均：exact=2/3, substitute=2/3, complement=0, irrelevant=0
    assert abs(m["macro_f1"] - (2 / 3 + 2 / 3) / 4) < 1e-9


def test_evaluate_records_splits_by_task():
    rows = list(read_jsonl(SAMPLE))
    outs = [format_output(r["label"], task_of(r)) for r in rows]
    report = evaluate_records(rows, outs)
    assert set(report) == {"full", "relevance"}
    assert report["relevance"]["accuracy"] == 1.0
    assert report["full"]["intent_acc"] == 1.0


def test_bootstrap_identical_systems_have_zero_diff():
    from ecomq.eval.compare import bootstrap

    groups = {f"q{i}": [("exact", {"a": "exact", "b": "exact"}), ("substitute", {"a": "exact", "b": "exact"})] for i in range(10)}
    out = bootstrap(groups, ["a", "b"], n_boot=50, seed=0)
    assert out["systems"]["a"]["accuracy"] == 0.5
    assert out["diffs"]["b - a"]["accuracy"] == 0.0
    assert out["diffs"]["b - a"]["macro_f1_ci95"] == [0.0, 0.0]
