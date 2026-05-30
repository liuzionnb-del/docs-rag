"""本地 bge-reranker 交叉编码器：对召回结果做精排（无需 API）。"""
from functools import lru_cache
from sentence_transformers import CrossEncoder
from config import RERANK_MODEL


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


def rerank(query: str, docs: list, top_n: int):
    """对 (query, doc) 打分排序，返回 [(doc, score), ...] 前 top_n 个。"""
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = _model().predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [(d, float(s)) for d, s in ranked[:top_n]]
