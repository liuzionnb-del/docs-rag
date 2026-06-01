"""本地 bge embedding 封装，实现 LangChain Embeddings 接口（无需 API）。"""
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from config import EMBED_MODEL


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


class BgeEmbeddings(Embeddings):
    """bge 系列建议对 query 加指令前缀以提升检索效果。"""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = _model().encode(texts, normalize_embeddings=True, batch_size=32)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = _model().encode(self.QUERY_PREFIX + text, normalize_embeddings=True)
        return vec.tolist()
