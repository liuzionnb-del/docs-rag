"""合成 65 道 in-scope gold 题。
用法:
  python eval/synth_inscope.py --dry   # 只抽样，不调 LLM(检查分布)
  python eval/synth_inscope.py         # 完整运行(抽样 + LLM 出题)

流程：
  1. 从 BM25 pickle 加载 3282 个已切分 chunk
  2. 按 doc_type 分层抽样(tutorial 21/advanced 15/reference 10/guide 9/concept 10)
     - 每文件 ≤ 1 道(最大化文件覆盖)
     - concept 过滤垃圾文件
     - 优先选 meaty chunk(200-1500字符,有 header,代码比 < 40%)
  3. LLM 出题(强约束 prompt,避免答案泄漏)
  4. 质量过滤(长度/banned phrase/词重叠)
  5. 合并 8 道已有 OOS,写入 gold_set_v2.json

复现性: random.seed(42) 固定。
"""
import json
import pickle
import random
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import HumanMessage, SystemMessage
from config import BM25_PATH
from llm import get_llm

# ── 常量 ──────────────────────────────────────────────────
SEED = 42
PLAN_PATH = ROOT / "eval" / "synth_plan.json"
GOLD_PATH = ROOT / "eval" / "gold_set_v2.json"

TARGETS = {"tutorial": 21, "advanced": 15, "reference": 10, "guide": 9, "concept": 10}

CONCEPT_EXCLUDE = {
    # 变更日志/贡献者/元信息
    "release-notes.md", "fastapi-people.md", "contributing.md",
    "external-links.md", "benchmarks.md", "editor-support.md",
    # 站务/UI/翻译/测试,不是 FastAPI 技术概念
    "newsletter.md", "management.md", "help-fastapi.md",
    "translation-banner.md", "translations.md", "_llm-test.md",
}

CODE_BLOCK_RE = re.compile(r"```.*?```", re.S)

BANNED_PHRASES = [
    "according to", "in the passage", "in the doc", "as mentioned",
    "as shown", "in the above", "this passage", "this section",
]


# ── meaty chunk 判定 ──────────────────────────────────────
def is_meaty(chunk: dict) -> bool:
    text = chunk["page_content"]
    if not (200 <= len(text) <= 1500):
        return False
    if not chunk["metadata"].get("header"):
        return False
    code_chars = sum(len(m) for m in CODE_BLOCK_RE.findall(text))
    if code_chars / max(len(text), 1) > 0.4:
        return False
    return True


# ── 分层抽样 ──────────────────────────────────────────────
def sample_slots(docs: list) -> list:
    random.seed(SEED)
    by_type: dict[str, list] = {}
    for d in docs:
        by_type.setdefault(d["metadata"]["doc_type"], []).append(d)

    slots = []
    for dtype, n in TARGETS.items():
        candidates = by_type.get(dtype, [])
        by_file: dict[str, list] = {}
        for c in candidates:
            by_file.setdefault(c["metadata"]["source"], []).append(c)
        files = sorted(by_file.keys())
        if dtype == "concept":
            files = [f for f in files if Path(f).name not in CONCEPT_EXCLUDE]
        if len(files) < n:
            print(f"  warn: {dtype} only {len(files)} eligible files (need {n})")
            n = len(files)
        chosen_files = random.sample(files, n)
        for f in chosen_files:
            meaty = [c for c in by_file[f] if is_meaty(c)]
            chunk = random.choice(meaty if meaty else by_file[f])
            slots.append({
                "doc_type": dtype,
                "file": f,
                "header": chunk["metadata"].get("header", ""),
                "content": chunk["page_content"],
            })
    return slots


# ── LLM 出题 ──────────────────────────────────────────────
SYS_PROMPT = (
    "You are building an evaluation set for a FastAPI documentation Q&A system. "
    "Given a passage from FastAPI docs, generate ONE question that:\n"
    "1. A FastAPI user might naturally ask BEFORE seeing this passage.\n"
    "2. Is answerable using only this passage's content.\n"
    "3. Uses general language — do NOT echo specific code symbols, identifiers, or quoted phrases from the passage.\n"
    "4. Is concise: one sentence, under 25 words.\n"
    "5. Does NOT use meta-phrases like 'according to', 'in the passage', 'as shown'.\n"
    'Output ONLY a JSON object: {"question": "..."}'
)


def gen_question(llm, slot: dict) -> str | None:
    user = (
        f"Source file: {slot['file']}\n"
        f"Header: {slot['header']}\n"
        f"Passage:\n---\n{slot['content']}\n---"
    )
    msg = llm.invoke([SystemMessage(content=SYS_PROMPT), HumanMessage(content=user)])
    m = re.search(r"\{.*\}", msg.content, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0)).get("question", "").strip()
    except Exception:
        return None


def quality_ok(q: str | None, slot: dict) -> tuple[bool, str]:
    if not q or not (15 <= len(q) <= 200):
        return False, f"len={len(q) if q else 0}"
    ql = q.lower()
    if any(p in ql for p in BANNED_PHRASES):
        return False, "banned"
    # 词级重叠泄漏检查
    chunk_lower = slot["content"].lower()
    words = re.findall(r"[a-z]{4,}", ql)
    if words:
        overlap = sum(1 for w in words if w in chunk_lower) / len(words)
        if overlap > 0.95:  # 仅挡近 copy-paste；技术词汇与文档共词正常
            return False, f"leak={overlap:.2f}"
    return True, "ok"


# ── 主流程 ────────────────────────────────────────────────
def main():
    payload = pickle.load(open(BM25_PATH, "rb"))
    docs = payload["docs"]
    print(f"loaded {len(docs)} chunks from {BM25_PATH.name}")

    slots = sample_slots(docs)
    by_type_count = {}
    for s in slots:
        by_type_count[s["doc_type"]] = by_type_count.get(s["doc_type"], 0) + 1
    print(f"sampled {len(slots)} slots, per-type: {by_type_count}")

    PLAN_PATH.write_text(json.dumps([
        {"doc_type": s["doc_type"], "file": s["file"],
         "header": s["header"][:60], "preview": s["content"][:120]}
        for s in slots
    ], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"plan -> {PLAN_PATH.name}\n")

    if "--dry" in sys.argv:
        print("--dry: skipping LLM generation. Inspect synth_plan.json.")
        return

    # 增量续跑：跳过已经在 gold_set_v2.json 里有对应文件的 slot
    existing_all = json.loads(GOLD_PATH.read_text(encoding="utf-8")) if GOLD_PATH.exists() else []
    existing_inscope = [e for e in existing_all if e.get("type") == "single"]
    covered_files = {e["expected_sources"][0] for e in existing_inscope if e.get("expected_sources")}
    if covered_files:
        print(f"existing in-scope: {len(existing_inscope)} entries covering {len(covered_files)} files (will skip)\n")

    llm_lo = get_llm(temperature=0.4)
    llm_hi = get_llm(temperature=0.7)
    entries, rejected = list(existing_inscope), []
    new_count = 0

    for i, slot in enumerate(slots, 1):
        if slot["file"] in covered_files:
            continue
        q = gen_question(llm_lo, slot)
        ok, why = quality_ok(q, slot)
        if not ok:
            q = gen_question(llm_hi, slot)  # 重试一次
            ok, why = quality_ok(q, slot)
        if not ok:
            rejected.append({"i": i, "file": slot["file"], "reason": why, "q": q})
            print(f"  [{i:>2}] {slot['doc_type']:<9} REJECT({why}) {slot['file']}")
            continue
        entries.append({
            "id": f"I{i:02d}",
            "type": "single",
            "doc_type": slot["doc_type"],
            "question": q,
            "expected_sources": [slot["file"]],
            "slot_meta": {
                "header": slot["header"][:80],
                "chunk_preview": slot["content"][:200],
            },
            "verified": False,
        })
        new_count += 1
        print(f"  [{i:>2}] {slot['doc_type']:<9} {slot['file']}")
        print(f"       -> {q}")

    print(f"\naccepted {new_count} new (total in-scope now {len(entries)}), rejected {len(rejected)}")

    # 合并 OOS
    oos = [e for e in existing_all if "oos" in e.get("type", "")]
    final = entries + oos
    GOLD_PATH.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ngold_set_v2.json: {len(entries)} in-scope + {len(oos)} oos = {len(final)} total")

    if rejected:
        print(f"\n--- rejected ({len(rejected)}) ---")
        for r in rejected:
            print(f"  [{r['i']:>2}] {r['reason']:<15} {r['file']}")
            print(f"       q: {r['q']}")


if __name__ == "__main__":
    main()
