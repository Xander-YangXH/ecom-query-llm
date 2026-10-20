import json

from ecomq.data.make_smoke import expand
from ecomq.data.io import read_jsonl
from ecomq.data.to_alpaca import convert, export
from ecomq.schema import build_user_prompt, task_of

SAMPLE = "data_sample/seed_sample.jsonl"


def test_expand_makes_unique_ids():
    rows = list(read_jsonl(SAMPLE))
    out = expand(rows, 40)
    assert len(out) == 40
    assert len({r["id"] for r in out}) == 40


def test_inference_prompt_matches_training_prompt():
    """LLaMA-Factory 把 instruction 与 input 用换行拼成同一个 user 消息；推理必须一致。"""
    for rec in read_jsonl(SAMPLE):
        row = convert(rec)
        assert build_user_prompt(rec["query"], rec.get("title"), task_of(rec)) == row["instruction"] + "\n" + row["input"]


def test_export_writes_dataset_info(tmp_path):
    split = tmp_path / "splits"
    split.mkdir()
    rows = list(read_jsonl(SAMPLE))
    for name in ("train", "val"):
        with open(split / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = export(split, tmp_path / "lf")
    assert set(info) == {"ecom_query_train", "ecom_query_val"}
    on_disk = json.loads((tmp_path / "lf" / "dataset_info.json").read_text(encoding="utf-8"))
    assert on_disk == info
    assert len(json.loads((tmp_path / "lf" / "ecom_query_train.json").read_text(encoding="utf-8"))) == len(rows)
