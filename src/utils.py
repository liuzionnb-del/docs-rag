"""共享工具：文档类型判定、文本清洗、BM25 分词。"""
import re
from pathlib import Path


def doc_type_for(relpath: str) -> str:
    """按文件相对路径的首段目录判定文档类型。"""
    top = Path(relpath).parts[0] if Path(relpath).parts else ""
    if top == "tutorial":
        return "tutorial"
    if top == "advanced":
        return "advanced"
    if top == "reference":
        return "reference"
    if top in ("deployment", "how-to"):
        return "guide"
    return "concept"  # 顶层 + about/learn


def clean_text(text: str) -> str:
    """折叠多余空行、去首尾空白。保留代码块缩进。"""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """BM25 分词：小写 + 提取字母数字下划线词（适合 API/代码术语）。"""
    return _TOKEN_RE.findall(text.lower())
