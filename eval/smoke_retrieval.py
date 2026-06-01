"""检索冒烟测试：相关问题应返回结果，无关问题应触发拒答。"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retrieval import HybridRetriever

r = HybridRetriever()

for q in [
    "How to add query parameter validation?",
    "How do I handle CORS in FastAPI?",
    "How to bake a chocolate cake?",  # 无关 -> 期望拒答
]:
    res, rejected = r.retrieve(q)
    print(f"\nQ: {q}")
    print(f"   rejected={rejected}")
    for d, s in res:
        print(f"   {s:6.2f} | {d.metadata['source']} | {d.metadata['header'][:40]}")
