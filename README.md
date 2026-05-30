# FastAPI Docs RAG 助手

面向开发者的 FastAPI 官方文档问答助手。围绕**离线索引**、**在线检索**、**效果评测**完成 RAG 全链路工程优化。

## 系统架构

```
                    ┌──────────────── 在线链路 (Online) ─────────────────┐
                    │                                                    │
   User question -> Query Rewrite ─┐                                     │
                                   │                                     │
                                   ▼                                     │
                    ┌──── Hybrid Retrieval ────┐                         │
                    │  · BM25 关键词召回 (k=10) │                         │
                    │  · FAISS 稠密召回 (k=10)  │                         │
                    └──── merge & dedup ───────┘                         │
                                   │                                     │
                                   ▼                                     │
                          bge-reranker 精排                              │
                                   │                                     │
                                   ▼                                     │
                       低置信度拒答 (score<0.3)                          │
                                   │                                     │
                                   ▼                                     │
                  强约束 Prompt + 引用生成 (LLM)                         │
                                   │                                     │
                                   ▼                                     │
                          Answer + Sources [n]                           │
                    └─────────────────────────────────────────────────────┘

                    ┌──────────────── 离线链路 (Offline) ──────────────┐
                    │  Markdown 文档 (152 个,5 类)                     │
                    │    │                                              │
                    │    ▼  按类型差异化切分 (tutorial/advanced/...)    │
                    │  3282 个 chunk (清洗 + 元数据绑定)               │
                    │    │                                              │
                    │    ├─→ bge-small-en-v1.5 → FAISS 索引            │
                    │    └─→ rank_bm25 关键词索引                       │
                    └──────────────────────────────────────────────────┘
```

四阶段架构:**Retriever → Reranker → Generator → Critic(拒答+引用校验)**

## 技术栈

| 层 | 选型 |
|----|------|
| 大模型(生成 + 改写 + 评估) | DeepSeek / 通义 / 智谱(兼容 OpenAI 接口,可配置) |
| Embedding | `BAAI/bge-small-en-v1.5`(本地,免 API) |
| Reranker | `BAAI/bge-reranker-base`(本地交叉编码器) |
| 向量索引 | **FAISS** |
| 关键词索引 | **BM25** (`rank_bm25`) |
| 切分 | `langchain-text-splitters` (MarkdownHeader + Recursive,按文档类型差异化) |
| API 服务 | **FastAPI** + Uvicorn |

## 工程亮点

### 离线索引优化
- 按 `tutorial / advanced / reference / guide / concept` **五种文档类型**分别设置 chunk 大小与重叠
- 先 Markdown 标题切分(保留 H1>H2>H3 层级到 metadata),再字符级 Recursive 二次切分
- 每个 chunk 绑定 `{source, doc_type, header}` 元数据,支持溯源
- BM25 + 稠密向量**双索引同步构建**

### 在线检索优化
- **查询改写**:LLM 将用户口语化问题改写为检索友好查询;失败自动退回原 query
- **多路召回**:BM25 + FAISS 各取 top-10,merge + 去重
- **bge-reranker 精排**:对召回候选交叉编码打分
- **低置信度拒答**:rerank 最高分 < 0.3 触发"知识库无此信息",防幻觉
- **强约束 Prompt**:明确只能基于编号上下文回答,要求带 `[n]` 引用标号

### 评估体系
- **自建 gold set**(20 题:17 in-scope + 3 out-of-scope)
- **检索指标**:Hit@K、MRR、Context Precision、Context Recall、拒答召回率
- **生成指标**:LLM-as-a-Judge 在 0–1 间评分 faithfulness / relevancy / citation
- **消融实验**:对比有无 rerank、有无查询改写

## 效果数据

### 检索评估(gold set, top_k=4)

| 配置 | Hit@4 | MRR | Ctx Precision | Ctx Recall | 拒答召回率 | 误拒率 |
|------|-------|-----|---------------|------------|------------|--------|
| **混合召回 + rerank(全链路)** | **1.000** | **0.912** | **0.574** | **0.971** | **1.000** | 0.000 |
| 消融:无 rerank | 0.941 | 0.799 | 0.500 | 0.941 | 0.000 | 0.000 |

**结论**:rerank 同时提升排序质量(MRR +14%)与拒答能力(0 → 100%),证明精排环节不可或缺。

### 生成评估(LLM-as-a-Judge, n=17 in-scope)

| 指标 | 平均分(0–1) |
|------|--------------|
| Faithfulness(无幻觉) | **0.897** |
| Relevancy(切题) | **0.882** |
| Citation(引用正确) | **0.824** |

12/17 完美得分,LLM-as-a-Judge 主动暴露 2 类缺陷:
- **检索质量不足导致的幻觉**(如 Q9:在没有"FastAPI testing convenience tool"的上下文里编造了一个工具)
- **答非所问**(如 Q7:回答内容忠实于上下文,但未回应用户实际意图)

→ 评估驱动后续优化方向:针对幻觉加强 retrieval 的 query expansion,针对答非所问优化生成 prompt 与上下文压缩。

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. 配置大模型 key
copy .env.example .env          # 填入 LLM_API_KEY

# 3. 准备语料(FastAPI 官方文档)
git clone --depth 1 https://github.com/fastapi/fastapi.git /tmp/fastapi
# 复制 /tmp/fastapi/docs/en/docs 下的 .md 到 data/raw/(保留子目录)

# 4. 构建离线索引
python src/build_index.py

# 5. 单条问答
python src/rag.py "How do I declare an optional query parameter?"

# 6. 启动 API 服务
uvicorn api:app --app-dir src --reload   # http://localhost:8000/docs

# 7. 跑评估
python eval/eval_retrieval.py            # 检索指标(无需 key)
python eval/eval_judge.py                # 生成指标(需 key)
```

## 目录结构

```
docs-rag/
├── data/raw/             FastAPI 文档语料(.md)
├── src/
│   ├── config.py         配置(从 .env 读取)
│   ├── utils.py          文档类型判定、清洗、BM25 分词
│   ├── embeddings.py     bge 嵌入封装
│   ├── reranker.py       bge-reranker 精排
│   ├── build_index.py    阶段1:离线索引构建
│   ├── retrieval.py      阶段2:混合召回 + rerank + 拒答
│   ├── llm.py            LLM 客户端(OpenAI 兼容)
│   ├── rag.py            阶段2:查询改写 + 引用生成
│   └── api.py            阶段4:FastAPI /ask 接口
├── eval/
│   ├── gold_set.json     20 题 gold set
│   ├── eval_retrieval.py 检索指标 + 消融
│   ├── eval_judge.py     LLM-as-a-Judge 生成评估
│   └── *_results.json    评估结果
└── storage/              生成的索引(FAISS + BM25)
```

## 工程过程中的真实问题与决策

- **chromadb 在目标机器上原生库段错误** → 改用 FAISS,更主流且兼容性好
- **torch 2.12 CPU 轮子 DLL 加载失败**(VC++ 运行库不匹配)→ 降级到稳定的 torch 2.2.2 + numpy<2
- **RAGAS 与新版 langchain-community 不兼容**(import 已移除的 `chat_models.vertexai`) → 改自建评估体系 + LLM-as-a-Judge,更可控也更贴现代实践
