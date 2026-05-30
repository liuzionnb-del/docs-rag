"""阶段3 生成质量评估：LLM-as-a-Judge。
对每条 in-scope 问题跑完整 RAG (检索+生成) -> 让 LLM 在 0-1 之间评分:
  - faithfulness:  回答是否完全基于给定上下文（无幻觉）
  - relevancy:     回答是否切题
  - citation:      引用标号 [n] 是否对应到实际引用了的上下文
需要 .env 配置 LLM_API_KEY 才能运行。
"""
import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import SystemMessage, HumanMessage
from rag import RagPipeline
from llm import get_llm

GOLD_PATH = ROOT / "eval" / "gold_set.json"
OUT_PATH = ROOT / "eval" / "judge_results.json"

JUDGE_SYSTEM = (
    "You are a strict evaluator for a RAG question-answering system. "
    "Given the question, the retrieved context (with source markers [n]), and the system's answer, "
    "score the answer on three dimensions in [0.0, 1.0] (use 0.25 steps):\n"
    "1. faithfulness: 1.0 if every claim is supported by the retrieved context, 0 if hallucinated.\n"
    "2. relevancy: 1.0 if the answer directly addresses the question, 0 if off-topic.\n"
    "3. citation: 1.0 if the answer uses bracket citations [n] that match the supporting context entries, "
    "0 if citations are missing/wrong.\n\n"
    "Respond ONLY with a JSON object: "
    "{\"faithfulness\": x, \"relevancy\": y, \"citation\": z, \"comment\": \"one sentence\"}."
)


def _parse_judge(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"faithfulness": 0.0, "relevancy": 0.0, "citation": 0.0, "comment": "parse_error"}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"faithfulness": 0.0, "relevancy": 0.0, "citation": 0.0, "comment": "json_error"}


def judge(question: str, answer: str, sources: list[dict]) -> dict:
    ctx_text = "\n".join(f"[{s['id']}] (source: {s['source']})" for s in sources)
    user = (
        f"Question: {question}\n\n"
        f"Retrieved context entries:\n{ctx_text}\n\n"
        f"System answer:\n{answer}\n\n"
        "Score now."
    )
    msg = get_llm(temperature=0.0).invoke([
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=user),
    ])
    return _parse_judge(msg.content.strip())


def main():
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    in_scope = [g for g in gold if g["type"] != "oos"]
    pipe = RagPipeline()

    results = []
    for g in in_scope:
        out = pipe.answer(g["question"])
        if out["rejected"]:
            results.append({"id": g["id"], "rejected": True,
                            "faithfulness": None, "relevancy": None, "citation": None})
            print(f"  Q{g['id']:>2}  REJECTED")
            continue
        scores = judge(g["question"], out["answer"], out["sources"])
        scores["id"] = g["id"]
        scores["rejected"] = False
        results.append(scores)
        print(f"  Q{g['id']:>2}  faith={scores['faithfulness']:.2f} "
              f"rel={scores['relevancy']:.2f} cite={scores['citation']:.2f}  "
              f"// {scores.get('comment','')[:60]}")

    scored = [r for r in results if not r["rejected"]]
    if scored:
        avg = lambda k: round(sum(r[k] for r in scored) / len(scored), 3)
        summary = {
            "n_judged": len(scored),
            "avg_faithfulness": avg("faithfulness"),
            "avg_relevancy": avg("relevancy"),
            "avg_citation": avg("citation"),
        }
        print("\n=== LLM-as-a-Judge 汇总 ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    else:
        summary = {"n_judged": 0}

    OUT_PATH.write_text(
        json.dumps({"summary": summary, "details": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
