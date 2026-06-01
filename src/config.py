"""集中配置：从 .env 读取，供各模块统一引用。"""
from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# 数据与存储路径
RAW_DIR = ROOT / "data" / "raw"
STORAGE_DIR = ROOT / "storage"
FAISS_DIR = STORAGE_DIR / "faiss"
BM25_PATH = STORAGE_DIR / "bm25.pkl"

# 大模型（兼容 OpenAI 接口的国内厂商）
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# 本地模型
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# 检索参数
CHUNK_SETTINGS = {
    # 按文档类型差异化切分：(chunk_size, overlap)
    "tutorial": (800, 150),     # 教程：连贯叙述，块大些保留上下文
    "advanced": (800, 150),
    "reference": (500, 50),     # API 参考：条目独立，块小、重叠少
    "guide": (700, 120),        # deployment / how-to
    "concept": (600, 100),      # 顶层概念页
}
STORAGE_DIR.mkdir(exist_ok=True)
