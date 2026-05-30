"""阶段2 完整 RAG 链路：查询改写 -> 混合检索 -> 拒答判定 -> 强约束生成(带引用)。
需要在 .env 配置 LLM_API_KEY。
"""
from langchain_core.messages import SystemMessage, HumanMessage

from retrieval import HybridRetriever
from llm import get_llm

REWRITE_PROMPT = (
    "You rewrite a user question into a concise English search query optimized for "
    "retrieving FastAPI documentation. Output ONLY the rewritten query, no explanation."
)

ANSWER_SYSTEM = (
    "You are a FastAPI documentation assistant. Answer ONLY using the numbered context "
    "below. Cite the sources you use with bracket markers like [1], [2]. If the context "
    "does not contain the answer, reply exactly: \"I don't have enough information in the "
    "documentation to answer that.\" Do not invent APIs or behavior."
)

REJECT_MSG = "I don't have enough information in the documentation to answer that."


class RagPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()

    def rewrite(self, question: str) -> str:
        try:
            llm = get_llm()
            msg = llm.invoke([
                SystemMessage(content=REWRITE_PROMPT),
                HumanMessage(content=question),
            ])
            return msg.content.strip() or question
        except Exception:
            return question  # 改写失败则退回原始问题

    def answer(self, question: str, use_rewrite: bool = True, use_rerank: bool = True) -> dict:
        query = self.rewrite(question) if use_rewrite else question
        results, rejected = self.retriever.retrieve(query, rerank_on=use_rerank)

        if rejected:
            return {"answer": REJECT_MSG, "sources": [], "rejected": True, "query": query}

        context_blocks, sources = [], []
        for i, (doc, score) in enumerate(results, 1):
            src = doc.metadata.get("source", "?")
            context_blocks.append(f"[{i}] (source: {src})\n{doc.page_content}")
            sources.append({"id": i, "source": src, "score": round(score, 3),
                            "header": doc.metadata.get("header", "")})
        context = "\n\n".join(context_blocks)

        llm = get_llm()
        msg = llm.invoke([
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ])
        return {"answer": msg.content.strip(), "sources": sources,
                "rejected": False, "query": query}


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "How do I declare an optional query parameter?"
    out = RagPipeline().answer(q)
    print("Q:", q)
    print("rewritten:", out["query"])
    print("\nANSWER:\n", out["answer"])
    print("\nSOURCES:")
    for s in out["sources"]:
        print(f"  [{s['id']}] {s['source']} (score {s['score']}) {s['header'][:40]}")
