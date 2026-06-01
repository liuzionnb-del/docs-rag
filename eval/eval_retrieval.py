"""阶段3 检索评估：Hit@k、MRR、拒答准确率。
对 in-scope 问题统计命中率，对 oos 问题统计拒答率。
支持消融实验：分别评估 rerank on/off、rewrite on/off（需 key）。

用法:
  python eval/eval_retrieval.py                    # 仅检索消融（无需 key）
  python eval/eval_retrieval.py --with-rewrite     # 加上查询改写对比（需 LLM key）
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retrieval import HybridRetriever

GOLD_PATH = ROOT / "eval" / "gold_set.json"
TOP_K = 4


def _hit_rank(retrieved_sources: list[str], expected: list[str]) -> int:
    """返回首个命中位置 (1-based)，未命中返回 0。"""
    for i, s in enumerate(retrieved_sources, 1):
        if s in expected:
            return i
    return 0


def evaluate(setting: str, retriever: HybridRetriever, gold: list[dict], rewriter=None) -> dict:
    in_scope = [g for g in gold if g["type"] != "oos"]
    oos = [g for g in gold if g["type"] == "oos"]
    hits, rranks, false_rejects = 0, [], 0
    correct_rejects = 0
    precisions, recalls = [], []
    rows = []

    for g in gold:
        q = g["question"]
        query = rewriter(q) if rewriter else q
        results, rejected = retriever.retrieve(query, top_n=TOP_K, rerank_on=("no_rerank" not in setting))
        srcs = [d.metadata["source"] for d, _ in results]
        if g["type"] == "oos":
            if rejected:
                correct_rejects += 1
            rows.append((g["id"], "oos", rejected, "-", srcs[:2]))
        else:
            if rejected:
                false_rejects += 1
                rows.append((g["id"], "in", "REJECT!", 0, srcs[:2]))
                continue
            rank = _hit_rank(srcs, g["expected_sources"])
            if rank > 0:
                hits += 1
                rranks.append(1 / rank)
            else:
                rranks.append(0.0)
            # context_precision: 检索 chunk 中相关的比例（同源多chunk都算相关）
            # context_recall: 期望来源中被检索覆盖到的比例（按唯一文件算，避免重复）
            expected_set = set(g["expected_sources"])
            relevant_chunks = sum(1 for s in srcs if s in expected_set)
            covered_sources = len(set(srcs) & expected_set)
            precisions.append(relevant_chunks / len(srcs) if srcs else 0.0)
            recalls.append(covered_sources / len(expected_set) if expected_set else 0.0)
            rows.append((g["id"], "in", False, rank or "MISS", srcs[:2]))

    hit_at_k = hits / len(in_scope) if in_scope else 0.0
    mrr = sum(rranks) / len(in_scope) if in_scope else 0.0
    ctx_precision = sum(precisions) / len(precisions) if precisions else 0.0
    ctx_recall = sum(recalls) / len(recalls) if recalls else 0.0
    reject_recall = correct_rejects / len(oos) if oos else 0.0
    false_reject_rate = false_rejects / len(in_scope) if in_scope else 0.0

    return {
        "setting": setting,
        "Hit@K": round(hit_at_k, 3),
        "MRR": round(mrr, 3),
        "Ctx_Precision": round(ctx_precision, 3),
        "Ctx_Recall": round(ctx_recall, 3),
        "Reject_Recall(oos)": round(reject_recall, 3),
        "False_Reject_Rate(in)": round(false_reject_rate, 3),
        "rows": rows,
    }


def _print_summary(results: list[dict]):
    print("\n" + "=" * 100)
    print(f"{'setting':<32}{'Hit@K':>8}{'MRR':>8}{'CtxPrec':>9}{'CtxRec':>9}{'RejRec':>9}{'FalseRej':>10}")
    print("-" * 100)
    for r in results:
        print(f"{r['setting']:<32}{r['Hit@K']:>8}{r['MRR']:>8}"
              f"{r['Ctx_Precision']:>9}{r['Ctx_Recall']:>9}"
              f"{r['Reject_Recall(oos)']:>9}{r['False_Reject_Rate(in)']:>10}")
    print("=" * 100)


def _print_per_query(result: dict):
    print(f"\n--- per-query [{result['setting']}] ---")
    for row in result["rows"]:
        qid, t, rej, rank, top = row
        print(f"  Q{qid:>2}  type={t:<4} rejected={str(rej):<7} rank={str(rank):<5} top: {top}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-rewrite", action="store_true", help="多跑一组带查询改写的评估（需要 LLM key）")
    parser.add_argument("--verbose", action="store_true", help="打印每条问题的命中详情")
    args = parser.parse_args()

    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    print(f"gold set: {len(gold)} questions "
          f"({sum(1 for g in gold if g['type']!='oos')} in-scope + "
          f"{sum(1 for g in gold if g['type']=='oos')} oos)\n")

    retriever = HybridRetriever()
    results = [
        evaluate("hybrid + rerank (full)", retriever, gold),
        evaluate("hybrid no_rerank (ablation)", retriever, gold),
    ]

    if args.with_rewrite:
        from rag import RagPipeline
        rp = RagPipeline()
        results.append(evaluate("hybrid + rerank + rewrite", retriever, gold, rewriter=rp.rewrite))

    if args.verbose:
        for r in results:
            _print_per_query(r)
    _print_summary(results)

    # 持久化
    out = ROOT / "eval" / "retrieval_results.json"
    out.write_text(json.dumps([{k: v for k, v in r.items() if k != "rows"} for r in results],
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
