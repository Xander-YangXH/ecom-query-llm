"""E↔S 混淆的标注审计（DEVLOG S17）：抽取系统性错误做盲评，估计标注本身有争议的比例。

sample  从测试集中抽样，写出盲评表（不含标注与模型预测，顺序打乱）和单独的密钥文件
  错误组：标注为 exact 而 ≥k 个模型判为 substitute（E2S），以及反过来（S2E）；每个 query 最多取 1 条
  对照组：所有模型都判对的 exact / substitute，用来衡量判断人自身与标注的一致程度
score   读入判断结果（item -> E/S/C/I/U，U 表示不确定），对照密钥计算各组一致率与“超出对照组的不一致率”
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ecomq.data.io import read_jsonl
from ecomq.schema import parse_output

ABBR = {"exact": "E", "substitute": "S", "complement": "C", "irrelevant": "I"}


def load_votes(pred_paths: list[str]) -> dict[str, list[str]]:
    votes = defaultdict(list)
    for path in pred_paths:
        for p in read_jsonl(path):
            obj = parse_output(p["output"], "relevance") or {}
            votes[p["id"]].append(obj.get("relevance", "invalid"))
    return votes


def select(recs, votes, k: int, n_err: int, n_ctrl: int, seed: int):
    rng = random.Random(seed)
    pools = defaultdict(list)
    for r in recs:
        gold, v = r["label"]["relevance"], votes[r["id"]]
        if gold == "exact" and v.count("substitute") >= k:
            pools["E2S"].append(r)
        elif gold == "substitute" and v.count("exact") >= k:
            pools["S2E"].append(r)
        elif gold in ("exact", "substitute") and all(x == gold for x in v):
            pools["control_" + ABBR[gold]].append(r)

    def pick(pool, n):
        rng.shuffle(pool)
        seen, out = set(), []
        for r in pool:  # 每个 query 最多 1 条，增加多样性
            if r["group"] not in seen:
                seen.add(r["group"])
                out.append(r)
            if len(out) == n:
                break
        return out

    chosen = {"E2S": pick(pools["E2S"], n_err), "S2E": pick(pools["S2E"], n_err),
              "control_E": pick(pools["control_E"], n_ctrl), "control_S": pick(pools["control_S"], n_ctrl)}
    sizes = {name: len(p) for name, p in pools.items()}
    items = [(name, r) for name, rs in chosen.items() for r in rs]
    rng.shuffle(items)
    return items, sizes


def score(key: dict, labels: dict[str, str]) -> dict:
    by_set = defaultdict(Counter)
    for item, info in key["items"].items():
        lab = labels.get(item)
        if lab is None:
            continue
        gold, model = info["gold"], info["model_majority"]
        if lab == "U":
            by_set[info["set"]]["uncertain"] += 1
        elif lab == gold:
            by_set[info["set"]]["agree_gold"] += 1
        elif lab == model:
            by_set[info["set"]]["agree_model"] += 1
        else:
            by_set[info["set"]]["other"] += 1
        by_set[info["set"]]["n"] += 1

    def rate(c, key_):
        return round(c[key_] / c["n"], 4) if c["n"] else None

    out = {name: {**dict(c), **{f"{k}_rate": rate(c, k) for k in ("agree_gold", "agree_model", "other", "uncertain")}}
           for name, c in sorted(by_set.items())}
    ctrl = by_set["control_E"] + by_set["control_S"]
    err = by_set["E2S"] + by_set["S2E"]
    if ctrl["n"] and err["n"]:
        ctrl_dis = 1 - ctrl["agree_gold"] / ctrl["n"]
        err_dis = 1 - err["agree_gold"] / err["n"]
        out["summary"] = {
            "error_items": err["n"], "control_items": ctrl["n"],
            "error_disagree_with_gold": round(err_dis, 4),
            "control_disagree_with_gold": round(ctrl_dis, 4),
            "excess_disagreement": round(err_dis - ctrl_dis, 4),
            "error_agree_with_model": round(err["agree_model"] / err["n"], 4),
            "error_uncertain": round(err["uncertain"] / err["n"], 4),
        }
    return out


def read_labels(path: str) -> dict[str, str]:
    """支持 CSV（列 item,label）或 Excel 盲评表（“标注”工作表的“序号”与“你的判断”列）。"""
    norm = {"E": "E", "S": "S", "C": "C", "I": "I", "U": "U", "?": "U", "不确定": "U"}
    if path.endswith(".xlsx"):
        import openpyxl

        ws = openpyxl.load_workbook(path, data_only=True)["标注"]
        header = [c.value for c in ws[1]]
        i_item, i_lab = header.index("序号"), header.index("你的判断")
        rows = ((r[i_item], r[i_lab]) for r in ws.iter_rows(min_row=2, values_only=True))
    else:
        rows = ((r["item"], r["label"]) for r in csv.DictReader(open(path, encoding="utf-8")))
    out = {}
    for item, lab in rows:
        lab = str(lab).strip().upper() if lab is not None else ""
        lab = norm.get(lab, norm.get(lab[:1], None)) if lab else None
        if item is not None and lab:
            out[str(int(item))] = lab
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--test", required=True)
    s.add_argument("--preds", nargs="+", required=True)
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--n-err", type=int, default=50)
    s.add_argument("--n-ctrl", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out-dir", required=True)
    c = sub.add_parser("score")
    c.add_argument("--key", required=True)
    c.add_argument("--labels", required=True)
    c.add_argument("--output")
    args = ap.parse_args()

    if args.cmd == "sample":
        recs = list(read_jsonl(args.test))
        votes = load_votes(args.preds)
        items, sizes = select(recs, votes, args.k, args.n_err, args.n_ctrl, args.seed)
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        key = {"k": args.k, "n_models": len(args.preds), "pool_sizes": sizes, "items": {}}
        with open(out / "blind.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item", "query", "title", "brand", "color", "details"])
            for i, (name, r) in enumerate(items, 1):
                p = r.get("product", {})
                w.writerow([i, r["query"], r["title"], p.get("brand", ""), p.get("color", ""), p.get("bullets", "")])
                v = votes[r["id"]]
                key["items"][str(i)] = {"id": r["id"], "set": name, "gold": ABBR[r["label"]["relevance"]],
                                        "model_majority": ABBR.get(Counter(v).most_common(1)[0][0], "?"),
                                        "votes": dict(Counter(ABBR.get(x, x) for x in v)),
                                        "query_id": r["meta"]["query_id"], "product_id": r["meta"]["product_id"]}
        (out / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"pool_sizes": sizes, "sampled": Counter(n for n, _ in items)}, ensure_ascii=False))
    else:
        report = score(json.loads(Path(args.key).read_text(encoding="utf-8")), read_labels(args.labels))
        text = json.dumps(report, indent=2, ensure_ascii=False)
        print(text)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
