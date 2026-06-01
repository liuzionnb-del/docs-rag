"""阶段4：FastAPI 服务，暴露 /ask /health /metrics 接口。
运行: uvicorn api:app --app-dir src --reload
依赖: pip install fastapi uvicorn
"""
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from rag import RagPipeline

_pipeline: RagPipeline | None = None
# 简易内存级 metrics
_metrics = {"requests": 0, "rejections": 0, "errors": 0, "latencies_ms": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时预热索引与模型，避免首请求 cold-start。"""
    global _pipeline
    print("[startup] warming up pipeline (FAISS + bge embedding + bge reranker)...")
    t0 = time.perf_counter()
    _pipeline = RagPipeline()
    print(f"[startup] ready in {time.perf_counter() - t0:.1f}s")
    yield
    print("[shutdown] bye")


app = FastAPI(title="FastAPI Docs RAG", version="0.1.0", lifespan=lifespan)


def _get_pipeline() -> RagPipeline:
    # 启动时已预热；保留懒加载兜底，避免单测/脚本场景失败
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
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
        if len(_metrics["latencies_ms"]) > 1000:
            _metrics["latencies_ms"] = _metrics["latencies_ms"][-1000:]
