"""阶段2 检索核心：多路召回(BM25+稠密) -> 合并去重 -> bge 重排 -> 拒答判定。
这部分全本地，不需要 API key。
"""
import pickle
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import FAISS_DIR, BM25_PATH
from embeddings import BgeEmbeddings
from utils import tokenize
from reranker import rerank

# 召回与精排参数
DENSE_K = 10          # 稠密召回数
BM25_K = 10           # 关键词召回数
TOP_N = 4             # 重排后保留数（最终送入生成的上下文条数）
REJECT_THRESHOLD = 0.3  # 重排最高分低于此值 -> 拒答（实测：相关问题 0.7+，无关问题 0.0，0.3 安全分割）


class HybridRetriever:
    def __init__(self):
        self._vs = FAISS.load_local(
            str(FAISS_DIR), BgeEmbeddings(), allow_dangerous_deserialization=True
        )
        with open(BM25_PATH, "rb") as f:
            payload = pickle.load(f)
        self._bm25 = payload["bm25"]
        self._bm25_docs = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in payload["docs"]
        ]

    def _dense(self, query: str, k: int) -> list[Document]:
        return self._vs.similarity_search(query, k=k)

    def _bm25_recall(self, query: str, k: int) -> list[Document]:
        scores = self._bm25.get_scores(tokenize(query))
        idx = np.argsort(scores)[::-1][:k]
        return [self._bm25_docs[i] for i in idx if scores[i] > 0]

    @staticmethod
    def _merge(*lists: list[Document]) -> list[Document]:
        seen, merged = set(), []
        for docs in lists:
            for d in docs:
                key = (d.metadata.get("source"), d.page_content[:80])
                if key not in seen:
                    seen.add(key)
                    merged.append(d)
        return merged

    def retrieve(self, query: str, top_n: int = TOP_N, rerank_on: bool = True):
        """返回 (results, rejected)。results=[(doc, score), ...]。
        rerank_on=False 时跳过精排（用于消融实验），按召回顺序取 top_n。
        """
        candidates = self._merge(
            self._dense(query, DENSE_K),
            self._bm25_recall(query, BM25_K),
        )
        if rerank_on:
            ranked = rerank(query, candidates, top_n)
            rejected = (not ranked) or (ranked[0][1] < REJECT_THRESHOLD)
        else:
            ranked = [(d, 0.0) for d in candidates[:top_n]]
            rejected = not ranked
        return ranked, rejected
