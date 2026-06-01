"""阶段3-v2 检索评估:
- 数据源: gold_set_v2.json (73 道,含 split / doc_type / hard_category 字段)
- 报表: global + per-doc_type + hard-oos 三层指标
- 数据集: --split test (默认报告) / tuning (调参用) / all
- 消融: rerank on/off,可选叠加 query rewrite (--with-rewrite, 需 LLM key)

用法:
  python eval/eval_retrieval_v2.py                      # test split, 不含改写
  python eval/eval_retrieval_v2.py --split tuning        # 调参用
  python eval/eval_retrieval_v2.py --with-rewrite        # 加一组带改写的对比
"""
import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retrieval import HybridRetriever

GOLD = ROOT / "eval" / "gold_set_v2.json"
OUT = ROOT / "eval" / "retrieval_results_v2.json"
TOP_K = 4


def _hit_rank(srcs: list[str], expected: list[str]) -> int:
    for i, s in enumerate(srcs, 1):
        if s in expected:
            return i
    return 0


def evaluate(setting: str, retriever: HybridRetriever, gold: list[dict], rewriter=None) -> dict:
    """返回 {setting, n, global, per_type:{...}, oos:{hard:{...}}}"""
    rerank_on = "no_rerank" not in setting

    # 按 stratum 累积原始量
    per_type_stats = defaultdict(lambda: {"hit": 0, "rranks": [], "prec": [], "rec": [],
                                          "false_reject": 0, "n": 0})
    oos_stats = {"hard": {"correct_reject": 0, "n": 0}}

    for g in gold:
        q = g["question"]
        query = rewriter(q) if rewriter else q
        results, rejected = retriever.retrieve(query, top_n=TOP_K, rerank_on=rerank_on)
        srcs = [d.metadata["source"] for d, _ in results]

        if "oos" in g["type"]:
            bucket = "hard"  # 现仅 hard-oos
            oos_stats[bucket]["n"] += 1
            if rejected:
                oos_stats[bucket]["correct_reject"] += 1
        else:
            t = g["doc_type"]
            s = per_type_stats[t]
            s["n"] += 1
            if rejected:
                s["false_reject"] += 1
                s["rranks"].append(0.0)
                s["prec"].append(0.0)
                s["rec"].append(0.0)
                continue
            rank = _hit_rank(srcs, g["expected_sources"])
            if rank > 0:
                s["hit"] += 1
                s["rranks"].append(1 / rank)
            else:
                s["rranks"].append(0.0)
            exp_set = set(g["expected_sources"])
            relevant = sum(1 for x in srcs if x in exp_set)
            covered = len(set(srcs) & exp_set)
            s["prec"].append(relevant / len(srcs) if srcs else 0)
            s["rec"].append(covered / len(exp_set) if exp_set else 0)

    # 汇总
    per_type = {}
    g_hit = g_rranks = g_prec = g_rec = 0
    g_total = 0
    for t, s in per_type_stats.items():
        n = s["n"]
        per_type[t] = {
            "n": n,
            "Hit@K": round(s["hit"] / n, 3),
            "MRR": round(sum(s["rranks"]) / n, 3),
            "Ctx_Prec": round(sum(s["prec"]) / n, 3),
            "Ctx_Rec": round(sum(s["rec"]) / n, 3),
            "False_Reject": round(s["false_reject"] / n, 3),
        }
        g_hit += s["hit"]
        g_rranks += sum(s["rranks"])
        g_prec += sum(s["prec"])
        g_rec += sum(s["rec"])
        g_total += n

    glb = {
        "n": g_total,
        "Hit@K": round(g_hit / g_total, 3) if g_total else 0,
        "MRR": round(g_rranks / g_total, 3) if g_total else 0,
        "Ctx_Prec": round(g_prec / g_total, 3) if g_total else 0,
        "Ctx_Rec": round(g_rec / g_total, 3) if g_total else 0,
    }

    oos = {}
    for bucket, s in oos_stats.items():
        if s["n"]:
            oos[bucket] = {
                "n": s["n"],
                "reject_rate": round(s["correct_reject"] / s["n"], 3),
            }

    return {"setting": setting, "global": glb, "per_type": per_type, "oos": oos}


def _print_report(results: list[dict]):
    for r in results:
        print(f"\n=== {r['setting']} ===")
        g = r["global"]
        print(f"GLOBAL  n={g['n']:<4}  Hit@K={g['Hit@K']:.3f}  MRR={g['MRR']:.3f}  "
              f"CtxPrec={g['Ctx_Prec']:.3f}  CtxRec={g['Ctx_Rec']:.3f}")
        print(f"\n  per-doc_type:")
        print(f"    {'type':<12}{'n':>4}{'Hit@K':>8}{'MRR':>8}{'CtxPrec':>9}{'CtxRec':>8}{'FalseRej':>10}")
        for t in sorted(r["per_type"]):
            p = r["per_type"][t]
            print(f"    {t:<12}{p['n']:>4}{p['Hit@K']:>8.3f}{p['MRR']:>8.3f}"
                  f"{p['Ctx_Prec']:>9.3f}{p['Ctx_Rec']:>8.3f}{p['False_Reject']:>10.3f}")
        if r["oos"]:
            print(f"\n  OOS reject rates:")
            for b, o in r["oos"].items():
                print(f"    {b:<6} n={o['n']:<3}  reject_rate={o['reject_rate']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "tuning", "all"], default="test",
                        help="数据集分片(默认 test,防数据泄漏)")
    parser.add_argument("--with-rewrite", action="store_true",
                        help="多跑一组带 LLM 查询改写的对比(需 key)")
    args = parser.parse_args()

    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    if args.split != "all":
        gold = [g for g in gold if g.get("split") == args.split]
    print(f"split={args.split}  n={len(gold)}  "
          f"(in-scope={sum(1 for g in gold if g['type']=='single')}, "
          f"hard-oos={sum(1 for g in gold if g['type']=='hard-oos')})")

    retriever = HybridRetriever()
    results = [
        evaluate("hybrid + rerank (full)", retriever, gold),
        evaluate("hybrid no_rerank (ablation)", retriever, gold),
    ]

    if args.with_rewrite:
        from rag import RagPipeline
        rp = RagPipeline()
        results.append(evaluate("hybrid + rerank + rewrite", retriever, gold, rewriter=rp.rewrite))

    _print_report(results)

    OUT.write_text(json.dumps({"split": args.split, "results": results},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
