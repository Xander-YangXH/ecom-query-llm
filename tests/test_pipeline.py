import json

from ecomq.data.io import read_jsonl
from ecomq.data.split import assign
from ecomq.data.to_alpaca import convert
from ecomq.data.validate import run
from ecomq.eval.metrics import evaluate
from ecomq.schema import format_output

SAMPLE = "data_sample/seed_sample.jsonl"


def test_validate_dedups_and_rejects():
    rows = list(read_jsonl(SAMPLE))
    dup = dict(rows[0], id="dup")
    bad = dict(rows[1], id="bad", label={**rows[1]["label"], "intent": "乱写"})
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


def test_to_alpaca_output_is_parseable_json():
    rec = next(iter(read_jsonl(SAMPLE)))
    row = convert(rec)
    assert json.loads(row["output"])["intent"] == rec["label"]["intent"]
    assert row["input"].startswith("query: ")


def test_metrics_perfect_and_broken():
    rows = list(read_jsonl(SAMPLE))
    golds = [r["label"] for r in rows]
    perfect = evaluate(golds, [format_output(g) for g in golds])
    assert perfect["format_valid_rate"] == 1.0
    assert perfect["intent_acc"] == 1.0
    assert perfect["attr_f1"] == 1.0
    assert perfect["relevance_acc"] == 1.0

    broken = evaluate(golds, ["garbage"] * len(golds))
    assert broken["format_valid_rate"] == 0.0
    assert broken["intent_acc"] == 0.0
    assert broken["attr_f1"] == 0.0
