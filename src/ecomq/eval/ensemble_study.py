"""S33 种子集成实验：已有打分的离线集成，不需要 GPU（设计与结论见 DEVLOG S33）。

协议（与单个模型完全相同）：成员打分按类平均（ensemble.average，默认 log p 平均）→ 在调参集上搜偏置 → 测试集报告。
每个系统报告：
- 先验调整后的 full / clean / merged 准确率与宏 F1、逐类 F1、raw（不加偏置）宏 F1；
- 与偏置无关的排序指标（complement / irrelevant 的 AP，E 对 S 的 AUC）。
clean 的争议集沿用 S18 标题版的委员会（E6 丰富输入三个种子的 raw 预测，三者都判反），所以 E6 不参与集成。

比较：
1. E2a 的 6 个种子（CCDS 3 个 + NSCC 3 个，同一配方）里任取 1/2/3/6 个做集成，比较各规模的范围（预注册标准：范围不重叠才算稳定）。
2. 按平台的 3 种子集成（CCDS、NSCC、CCDS-dq）与 N3、E10 的 3 种子集成：集成 vs 自己成员的均值，按 query 配对 bootstrap。
3. 跨配方集成（E2a + N3 / E10 / E8）与同样 6 个成员的 E2a 集成比较，看配方多样性是否比多一个种子更有用。
4. E2a-dq 三种子集成与 N1（14B，单模型）的报告范围比较（同一 double-quant 打分协议）。
  PYTHONPATH=src python -m ecomq.eval.ensemble_study --cache .cache --output reports/s33-ensembles.json
  PYTHONPATH=src python -m ecomq.eval.ensemble_study --cache .cache --n-bag 50 --output reports/s33-ensembles-bagged.json
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from ecomq.data.io import read_jsonl
from ecomq.eval.ensemble import average
from ecomq.eval.noise_aware import MERGE, confusion_scores, disputed_mask, load_preds
from ecomq.eval.prior_adjust import bagged_bias, macro_f1, predict, tune_bias
from ecomq.eval.ranking import ranking_metrics
from ecomq.schema import RELEVANCE

# 成员名 -> .cache/scores 下的文件前缀
MEMBERS = {
    "e2a-c42": "e2a", "e2a-c43": "e5-20k-s43", "e2a-c44": "e5-20k-s44",
    "e2a-n42": "n0-e2a-s42", "e2a-n43": "n0-e2a-s43", "e2a-n44": "n0-e2a-s44",
    "e2adq-42": "e2a-dq-s42", "e2adq-43": "e2a-dq-s43", "e2adq-44": "e2a-dq-s44",
    "n3-42": "n3-full-lr5e-6-s42", "n3-43": "n3-full-lr5e-6-s43", "n3-44": "n3-full-lr5e-6-s44",
    "e10-42": "e10-c1500-i1500-s42", "e10-43": "e10-c1500-i1500-s43", "e10-44": "e10-c1500-i1500-s44",
    "e8-42": "e8-c1500-s42", "e8-43": "e8-c1500-s43", "e8-44": "e8-c1500-s44",
}
E2A_POOL = ["e2a-c42", "e2a-c43", "e2a-c44", "e2a-n42", "e2a-n43", "e2a-n44"]
GROUPS = {
    "E2a-CCDS": ["e2a-c42", "e2a-c43", "e2a-c44"],
    "E2a-NSCC": ["e2a-n42", "e2a-n43", "e2a-n44"],
    "E2a-dq": ["e2adq-42", "e2adq-43", "e2adq-44"],
    "N3": ["n3-42", "n3-43", "n3-44"],
    "E10": ["e10-42", "e10-43", "e10-44"],
    "E8-c1500": ["e8-42", "e8-43", "e8-44"],
}
CROSS = {
    "E2a-6seeds": E2A_POOL,
    "E2a-NSCC+N3": GROUPS["E2a-NSCC"] + GROUPS["N3"],
    "E2a-CCDS+E10": GROUPS["E2a-CCDS"] + GROUPS["E10"],
    "E2a-CCDS+E8": GROUPS["E2a-CCDS"] + GROUPS["E8-c1500"],
    "all-15": E2A_POOL + GROUPS["N3"] + GROUPS["E10"],
}
COMMITTEE = ["e6-rich-s42", "e6-rich-s43", "e6-rich-s44"]


def per_class_f1(gold, pred):
    out = {}
    for c, name in enumerate(RELEVANCE):
        tp = np.sum((pred == c) & (gold == c))
        p, g = np.sum(pred == c), np.sum(gold == c)
        out[name] = float(2 * tp / (p + g)) if p + g else 0.0
    return out


class Study:
    def __init__(self, cache: Path, mode: str, n_bag: int = 0):
        self.mode, self.n_bag = mode, n_bag
        tune = list(read_jsonl(cache / "recs/tune.jsonl"))
        test = list(read_jsonl(cache / "recs/test.jsonl"))
        self.tune_ids, self.test_ids = [r["id"] for r in tune], [r["id"] for r in test]
        self.g_tune = np.array([RELEVANCE.index(r["label"]["relevance"]) for r in tune])
        self.g_test = np.array([RELEVANCE.index(r["label"]["relevance"]) for r in test])
        self.groups = np.array([r["group"] for r in test])
        self.tune_groups = np.array([r["group"] for r in tune])
        committee = [load_preds(str(cache / f"preds/{c}-raw.jsonl"), self.test_ids) for c in COMMITTEE]
        self.disputed = disputed_mask(self.g_test, committee)
        self.L = {}
        for name, prefix in MEMBERS.items():
            self.L[name] = (self._load(cache / f"scores/{prefix}-tune.jsonl", self.tune_ids),
                            self._load(cache / f"scores/{prefix}-test.jsonl", self.test_ids))
        self.preds = {}

    @staticmethod
    def _load(path, ids):
        by_id = {r["id"]: [r["logp"][c] for c in RELEVANCE] for r in read_jsonl(path)}
        return np.array([by_id[i] for i in ids], dtype=np.float64)

    def evaluate(self, members: list[str], mode: str | None = None) -> dict:
        mode = mode or self.mode
        Lt = average([self.L[m][0] for m in members], mode)
        Ls = average([self.L[m][1] for m in members], mode)
        if self.n_bag > 0:
            bias = bagged_bias(Lt, self.g_tune, self.tune_groups, self.n_bag)
            tune_after = macro_f1(self.g_tune, predict(Lt, bias))
        else:
            bias, tune_after = tune_bias(Lt, self.g_tune)
        g, raw, adj = self.g_test, predict(Ls, np.zeros(4)), predict(Ls, bias)
        keep = ~self.disputed
        key = "+".join(members) + ("" if mode == "logp" else f"@{mode}")
        self.preds[key] = adj
        rank = ranking_metrics(g, Ls)
        return {
            "members": members, "mode": mode, "n_members": len(members),
            "bias": dict(zip(RELEVANCE, map(float, bias))), "tune_after": tune_after,
            "raw": dict(zip(("acc", "f1"), confusion_scores(g, raw, 4))),
            "full": dict(zip(("acc", "f1"), confusion_scores(g, adj, 4))),
            "clean": dict(zip(("acc", "f1"), confusion_scores(g[keep], adj[keep], 4))),
            "merged": dict(zip(("acc", "f1"), confusion_scores(MERGE[g], MERGE[adj], 3))),
            "per_class_f1": per_class_f1(g, adj),
            "ranking": {"complement_ap": rank["complement"]["ap"], "irrelevant_ap": rank["irrelevant"]["ap"],
                        "exact_vs_sub_auc": rank["exact_vs_sub"]["auc"]},
        }

    def paired_bootstrap(self, members: list[str], n_boot: int, seed: int) -> dict:
        """集成（先验调整后）减去成员单独使用时的平均宏 F1，按 query 重采样；三种口径都算。"""
        ens = self.preds["+".join(members)]
        singles = [self.preds[m] for m in members]
        uniq, inv = np.unique(self.groups, return_inverse=True)
        rows_by_q = [np.flatnonzero(inv == q) for q in range(len(uniq))]
        g, keep = self.g_test, ~self.disputed

        def diff(rows):
            out = {}
            for view in ("full", "clean", "merged"):
                if view == "clean":
                    r = rows[keep[rows]]
                    f = lambda p: confusion_scores(g[r], p[r], 4)[1]
                elif view == "merged":
                    f = lambda p: confusion_scores(MERGE[g[rows]], MERGE[p[rows]], 3)[1]
                else:
                    f = lambda p: confusion_scores(g[rows], p[rows], 4)[1]
                out[view] = f(ens) - float(np.mean([f(p) for p in singles]))
            return out

        point = diff(np.arange(len(g)))
        rng = np.random.default_rng(seed)
        draws = [diff(np.concatenate([rows_by_q[q] for q in rng.integers(0, len(uniq), len(uniq))]))
                 for _ in range(n_boot)]
        return {v: {"diff": round(point[v], 4),
                    "ci95": [round(float(x), 4) for x in np.percentile([d[v] for d in draws], [2.5, 97.5])]}
                for v in point}


def span(rows, path):
    vals = []
    for r in rows:
        x = r
        for k in path:
            x = x[k]
        vals.append(x)
    return {"mean": round(float(np.mean(vals)), 4), "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4), "n": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=("logp", "prob"), default="logp")
    ap.add_argument("--n-bag", type=int, default=0, help="0：单次搜偏置（原协议）；>0：bagged 偏置（S33）")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    st = Study(Path(args.cache), args.mode, args.n_bag)

    singles = {m: st.evaluate([m]) for m in MEMBERS}
    report = {"mode": args.mode, "n_bag": args.n_bag, "disputed": {"n": int(st.disputed.sum()), "committee": COMMITTEE},
              "singles": singles, "e2a_pool": {}, "groups": {}, "cross": {}}

    # 1. E2a 6 个种子里任取 k 个
    paths = {"full_f1": ("full", "f1"), "clean_f1": ("clean", "f1"), "merged_f1": ("merged", "f1"),
             "full_acc": ("full", "acc"), "raw_f1": ("raw", "f1"), "complement_ap": ("ranking", "complement_ap"),
             "irrelevant_ap": ("ranking", "irrelevant_ap"), "exact_vs_sub_auc": ("ranking", "exact_vs_sub_auc")}
    for k in (1, 2, 3, 6):
        rows = [singles[c[0]] if k == 1 else st.evaluate(list(c)) for c in combinations(E2A_POOL, k)]
        report["e2a_pool"][f"k{k}"] = {name: span(rows, p) for name, p in paths.items()}

    # 2. 3 种子集成 vs 成员
    for name, members in GROUPS.items():
        ens = st.evaluate(members)
        prob = st.evaluate(members, "prob")
        report["groups"][name] = {
            "ensemble": ens,
            "ensemble_prob_mode": {k: prob[k] for k in ("full", "clean", "merged", "raw")},
            "members": {n: span([singles[m] for m in members], p) for n, p in paths.items()},
            "paired_bootstrap_vs_members": st.paired_bootstrap(members, args.n_boot, args.seed),
        }

    # 3. 跨配方集成
    for name, members in CROSS.items():
        report["cross"][name] = st.evaluate(members)

    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    def fmt(r):
        return (f"full {r['full']['f1']:.4f} acc {r['full']['acc']:.4f} clean {r['clean']['f1']:.4f} "
                f"merged {r['merged']['f1']:.4f} raw {r['raw']['f1']:.4f} | C-AP {r['ranking']['complement_ap']:.3f} "
                f"I-AP {r['ranking']['irrelevant_ap']:.3f} E/S-AUC {r['ranking']['exact_vs_sub_auc']:.3f}")
    print("== E2a pool (6 seeds), k members: mean [min, max]")
    for k, v in report["e2a_pool"].items():
        print(k, "  ".join(f"{n} {s['mean']:.4f} [{s['min']:.4f},{s['max']:.4f}]" for n, s in v.items()
                           if n in ("full_f1", "merged_f1", "clean_f1", "full_acc", "complement_ap")))
    print("== 3-seed groups")
    for name, v in report["groups"].items():
        m = v["members"]
        print(f"{name:10s} ens  {fmt(v['ensemble'])}")
        print(f"{'':10s} mem  full {m['full_f1']['mean']:.4f} [{m['full_f1']['min']:.4f},{m['full_f1']['max']:.4f}] "
              f"merged {m['merged_f1']['mean']:.4f} clean {m['clean_f1']['mean']:.4f}  boot {v['paired_bootstrap_vs_members']}")
        print(f"{'':10s} prob full {v['ensemble_prob_mode']['full']['f1']:.4f} merged {v['ensemble_prob_mode']['merged']['f1']:.4f}")
    print("== cross-recipe")
    for name, r in report["cross"].items():
        print(f"{name:14s} {fmt(r)}")


if __name__ == "__main__":
    main()
