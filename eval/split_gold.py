"""按 doc_type / hard-oos 分层切 30/70 -> tuning/test，写回 gold_set_v2.json 的 split 字段。

设计:
- in-scope 按 doc_type 分层
- hard-oos 整体作一层
- 每层独立 shuffle(seed=42)取前 30% 作 tuning
- 30/70 比例,标准四舍五入

工作流约束(在 README 写清):
- 调阈值/Prompt 时仅用 tuning 集
- 报告最终数字时仅用 test 集
- 评估脚本默认 --split test,防数据泄漏
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "eval" / "gold_set_v2.json"
SEED = 42
TUNING_RATIO = 0.30


def stratum_of(entry: dict) -> str:
    if entry["type"] == "single":
        return f"in:{entry['doc_type']}"
    if "oos" in entry["type"]:
        return entry["type"]
    return "other"


def main():
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    groups: dict[str, list] = defaultdict(list)
    for e in gold:
        groups[stratum_of(e)].append(e["id"])

    rng = random.Random(SEED)
    split_assignment: dict[str, str] = {}  # id -> "tuning"|"test"
    print(f"{'stratum':<20}{'total':>6}{'tuning':>8}{'test':>6}")
    print("-" * 40)

    for stratum in sorted(groups):
        ids = groups[stratum][:]
        rng.shuffle(ids)
        n_tuning = round(len(ids) * TUNING_RATIO)
        for i, eid in enumerate(ids):
            split_assignment[eid] = "tuning" if i < n_tuning else "test"
        print(f"{stratum:<20}{len(ids):>6}{n_tuning:>8}{len(ids)-n_tuning:>6}")

    # 写回
    for e in gold:
        e["split"] = split_assignment[e["id"]]

    GOLD.write_text(json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8")

    tuning_total = sum(1 for v in split_assignment.values() if v == "tuning")
    test_total = len(gold) - tuning_total
    print("-" * 40)
    print(f"{'TOTAL':<20}{len(gold):>6}{tuning_total:>8}{test_total:>6}")
    print(f"\nratio: tuning={tuning_total/len(gold):.1%}, test={test_total/len(gold):.1%}")
    print(f"\nsaved -> {GOLD}")


if __name__ == "__main__":
    main()
