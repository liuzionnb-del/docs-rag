"""阶段3-v2 生成质量评估:LLM-as-a-Judge
- 数据源: gold_set_v2.json,按 --split 过滤(默认 test)
- 仅对 in-scope 题跑 RAG -> 判分 faithfulness / relevancy / citation
- 输出: 全局均分 + per-doc_type 均分,落盘 judge_results_v2.json
"""
import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import HumanMessage, SystemMessage
from llm import get_llm
from rag import RagPipeline

GOLD = ROOT / "eval" / "gold_set_v2.json"
OUT = ROOT / "eval" / "judge_results_v2.json"

JUDGE_SYSTEM = (
    "You are a strict evaluator for a RAG question-answering system. "
    "Given the question, the retrieved context (with source markers [n]), and the system's answer, "
    "score in [0.0, 1.0] (0.25 steps):\n"
    "1. faithfulness: 1.0 if every claim is supported by retrieved context, 0 if hallucinated.\n"
    "2. relevancy: 1.0 if the answer directly addresses the question, 0 if off-topic.\n"
    "3. citation: 1.0 if [n] markers correctly point to supporting context, 0 if missing/wrong.\n\n"
    'Respond ONLY with JSON: {"faithfulness": x, "relevancy": y, "citation": z, "comment": "1 sentence"}.'
)


def _parse(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"faithfulness": 0.0, "relevancy": 0.0, "citation": 0.0, "comment": "parse_err"}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"faithfulness": 0.0, "relevancy": 0.0, "citation": 0.0, "comment": "json_err"}


def judge(question: str, answer: str, sources: list[dict]) -> dict:
    ctx = "\n".join(f"[{s['id']}] (source: {s['source']})" for s in sources)
    user = (f"Question: {question}\n\nRetrieved context:\n{ctx}\n\n"
            f"System answer:\n{answer}\n\nScore now.")
    msg = get_llm(temperature=0.0).invoke([
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=user),
    ])
    return _parse(msg.content.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "tuning", "all"], default="test")
    args = parser.parse_args()

    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    if args.split != "all":
        gold = [g for g in gold if g.get("split") == args.split]
    in_scope = [g for g in gold if g["type"] == "single"]
    print(f"split={args.split}  in-scope to judge: {len(in_scope)}\n")

    pipe = RagPipeline()
    results = []
    per_type = defaultdict(list)

    for i, g in enumerate(in_scope, 1):
        out = pipe.answer(g["question"])
        entry = {"id": g["id"], "doc_type": g["doc_type"], "split": g["split"]}
        if out["rejected"]:
            entry.update({"rejected": True, "faithfulness": None,
                          "relevancy": None, "citation": None})
            print(f"  [{i:>2}/{len(in_scope)}] Q{g['id']} {g['doc_type']:<10} REJECTED")
        else:
            sc = judge(g["question"], out["answer"], out["sources"])
            entry.update({"rejected": False, **sc})
            per_type[g["doc_type"]].append(sc)
            print(f"  [{i:>2}/{len(in_scope)}] Q{g['id']} {g['doc_type']:<10} "
                  f"faith={sc['faithfulness']:.2f} rel={sc['relevancy']:.2f} cite={sc['citation']:.2f}")
        results.append(entry)

    scored = [r for r in results if not r["rejected"]]
    summary = {"n_judged": len(scored), "n_rejected": len(results) - len(scored), "global": {}, "per_type": {}}
    if scored:
        for k in ("faithfulness", "relevancy", "citation"):
            summary["global"][k] = round(sum(r[k] for r in scored) / len(scored), 3)
    for t, lst in per_type.items():
        if lst:
            summary["per_type"][t] = {
                "n": len(lst),
                "faithfulness": round(sum(s["faithfulness"] for s in lst) / len(lst), 3),
                "relevancy": round(sum(s["relevancy"] for s in lst) / len(lst), 3),
                "citation": round(sum(s["citation"] for s in lst) / len(lst), 3),
            }

    print("\n=== summary ===")
    print(f"  judged={summary['n_judged']}, rejected={summary['n_rejected']}")
    g = summary["global"]
    if g:
        print(f"  GLOBAL  faith={g['faithfulness']:.3f}  rel={g['relevancy']:.3f}  cite={g['citation']:.3f}")
    print(f"  per-doc_type:")
    print(f"    {'type':<12}{'n':>4}{'faith':>8}{'rel':>8}{'cite':>8}")
    for t in sorted(summary["per_type"]):
        p = summary["per_type"][t]
        print(f"    {t:<12}{p['n']:>4}{p['faithfulness']:>8.3f}{p['relevancy']:>8.3f}{p['citation']:>8.3f}")

    OUT.write_text(json.dumps({"split": args.split, "summary": summary, "details": results},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
