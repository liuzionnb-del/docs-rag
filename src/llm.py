"""LLM 客户端：兼容 OpenAI 接口的国内厂商（DeepSeek/通义/智谱）。"""
from functools import lru_cache
from langchain_openai import ChatOpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    if not LLM_API_KEY:
        raise RuntimeError(
            "未配置 LLM_API_KEY。请复制 .env.example 为 .env 并填入你的 key。"
        )
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
        timeout=60,
    )
