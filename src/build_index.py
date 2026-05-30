"""阶段1：离线索引。
按文档类型差异化切分 -> 清洗 + 元数据绑定 -> 建 Chroma(稠密) + BM25(关键词) 混合索引。
运行: python src/build_index.py
"""
import pickle
import shutil

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from config import RAW_DIR, FAISS_DIR, BM25_PATH, CHUNK_SETTINGS
from embeddings import BgeEmbeddings
from utils import doc_type_for, clean_text, tokenize

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def load_and_split() -> list[Document]:
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
    chunks: list[Document] = []

    for path in sorted(RAW_DIR.rglob("*.md")):
        relpath = path.relative_to(RAW_DIR).as_posix()
        dtype = doc_type_for(relpath)
        size, overlap = CHUNK_SETTINGS[dtype]
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        if not text:
            continue

        # 先按 markdown 标题切，保留 section 层级到 metadata
        sections = md_splitter.split_text(text)
        # 再按类型对应的粒度做二次切分
        char_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        for sec in sections:
            header_path = " > ".join(
                sec.metadata[k] for k in ("h1", "h2", "h3") if sec.metadata.get(k)
            )
            for piece in char_splitter.split_text(sec.page_content):
                piece = piece.strip()
                if len(piece) < 30:  # 丢弃过短噪声块
                    continue
                chunks.append(Document(
                    page_content=piece,
                    metadata={"source": relpath, "doc_type": dtype, "header": header_path},
                ))
    return chunks


def build():
    print(f"[1/4] 读取并差异化切分 {RAW_DIR} ...")
    chunks = load_and_split()
    print(f"      共生成 {len(chunks)} 个 chunk")
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.metadata["doc_type"]] = by_type.get(c.metadata["doc_type"], 0) + 1
    print(f"      按类型分布: {by_type}")

    print("[2/4] 构建稠密向量索引 (FAISS + bge) ...")
    if FAISS_DIR.exists():
        shutil.rmtree(FAISS_DIR)
    vs = FAISS.from_documents(documents=chunks, embedding=BgeEmbeddings())
    vs.save_local(str(FAISS_DIR))

    print("[3/4] 构建 BM25 关键词索引 ...")
    tokenized = [tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    payload = {
        "bm25": bm25,
        "docs": [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks],
    }
    with open(BM25_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"[4/4] 完成。FAISS -> {FAISS_DIR}，BM25 -> {BM25_PATH}")


if __name__ == "__main__":
    build()
