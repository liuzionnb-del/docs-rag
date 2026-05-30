"""阶段4：FastAPI 服务，暴露 /ask /health /metrics 接口。
运行: uvicorn api:app --app-dir src --reload
依赖: pip install fastapi uvicorn
"""
import time
from fastapi import FastAPI
from pydantic import BaseModel

from rag import RagPipeline

app = FastAPI(title="FastAPI Docs RAG", version="0.1.0")
_pipeline: RagPipeline | None = None

# 简易内存级 metrics
_metrics = {"requests": 0, "rejections": 0, "errors": 0, "latencies_ms": []}


def _get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()  # 懒加载：首次请求时再加载索引/模型
    return _pipeline


class AskRequest(BaseModel):
    question: str
    use_rewrite: bool = True
    use_rerank: bool = True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    lat = _metrics["latencies_ms"]
    lat_sorted = sorted(lat)
    n = len(lat_sorted)
    return {
        "requests": _metrics["requests"],
        "rejections": _metrics["rejections"],
        "rejection_rate": round(_metrics["rejections"] / max(_metrics["requests"], 1), 3),
        "errors": _metrics["errors"],
        "latency_ms_p50": lat_sorted[n // 2] if n else 0,
        "latency_ms_p95": lat_sorted[int(n * 0.95)] if n else 0,
        "latency_ms_avg": round(sum(lat) / n, 1) if n else 0,
    }


@app.post("/ask")
def ask(req: AskRequest):
    t0 = time.perf_counter()
    _metrics["requests"] += 1
    try:
        out = _get_pipeline().answer(
            req.question, use_rewrite=req.use_rewrite, use_rerank=req.use_rerank
        )
        if out.get("rejected"):
            _metrics["rejections"] += 1
        return out
    except Exception as e:
        _metrics["errors"] += 1
        return {"error": str(e), "rejected": False, "answer": "", "sources": []}
    finally:
        _metrics["latencies_ms"].append(int((time.perf_counter() - t0) * 1000))
        # 只保留最近 1000 条，避免内存无界
        if len(_metrics["latencies_ms"]) > 1000:
            _metrics["latencies_ms"] = _metrics["latencies_ms"][-1000:]
