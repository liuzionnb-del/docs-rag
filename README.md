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
- **自建 gold set v2**(73 题:65 in-scope 按文档类型分层 + 8 hard-OOS 语义邻居)
- **30/70 tuning/test 分层切分**,防数据泄漏(阈值/Prompt 只在 tuning 调,test 仅出报告)
- **检索指标**:Hit@K、MRR、Context Precision、Context Recall、拒答召回率
- **生成指标**:LLM-as-a-Judge 在 0–1 间评分 faithfulness / relevancy / citation
- **per-type 分解**:每个 doc_type 独立报数,定位类别级问题
- **消融实验**:对比有无 rerank、有无查询改写

## 效果数据(v2 评估集,test split n=52)

### 检索评估(top_k=4)

**全局对比:**

| 配置 | Hit@4 | MRR | Ctx Precision | Ctx Recall | hard-OOS 拒答 |
|------|-------|-----|---------------|------------|----------------|
| **混合召回 + rerank(全链路)** | **0.826** | **0.670** | 0.370 | **0.826** | **0.500** |
| 消融:无 rerank | 0.761 | 0.630 | 0.370 | 0.761 | 0.000 |

**rerank 收益最大的是拒答**(0 → 50%):无 rerank 时模型对语义邻居 hard-OOS 完全无识别力,rerank 阈值是 Critic 层的核心机制。

**per-doc_type Hit@4 分解(全链路):**

| doc_type | n | Hit@4 (full) | Hit@4 (no rerank) | rerank Δ |
|----------|---|--------------|-------------------|----------|
| tutorial | 15 | **1.000** | 0.867 | +0.13 |
| guide | 6 | **1.000** | 0.833 | +0.17 |
| concept | 7 | 0.714 | 0.714 | 0 |
| reference | 7 | 0.714 | 0.571 | **+0.14** |
| advanced | 11 | 0.636 | 0.727 | **-0.09** ⚠ |

**两个结构性发现**:
- **reference 类受益最大**(+14pp Hit@4):验证了差异化切分中给 reference 用小 chunk(500/50)的设计——短小密集型在 bi-encoder 阶段信息丢失更严重,精排弥补关键
- **advanced 类 rerank 反而退步**(-9pp):误拒 1 道,提示当前 0.3 拒答阈值在 advanced 上偏严,后续可按类自适应

### 生成评估(LLM-as-a-Judge, test split n=46)

| | n | Faithfulness | Relevancy | Citation |
|---|---|--------------|-----------|----------|
| **GLOBAL** | 46 | **0.848** | **0.935** | **0.832** |
| guide | 6 | 1.000 | 1.000 | 1.000 |
| advanced | 11 | 0.932 | 0.818 | 0.932 |
| concept | 7 | 0.929 | 1.000 | 0.929 |
| reference | 7 | 0.786 | 1.000 | 0.786 |
| **tutorial** | 15 | **0.717** | 0.933 | **0.667** |

**反直觉发现**:tutorial 检索 Hit@4=100%,但**生成 faithfulness 仅 0.72、citation 仅 0.67**——证据充足时模型反而更敢"补内容"。后续要在 Prompt 里加更强的"仅基于上下文"约束、缩短 context 长度。

### 评估方法学

- **gold set 生成**:按 doc_type 分层抽样(tutorial 21 / advanced 15 / reference 10 / guide 9 / concept 10),每文件最多 1 道,LLM 在受控 prompt 下出"自然用户口吻"问题,期望来源由抽样自动绑定;启发式过滤泄漏(词重叠>0.95),1 道因"话题共通词"导致的过滤误报由人工放行
- **hard-OOS 设计**:8 道 4 类(同类竞品 / 不存在功能 / 关联底层 / 超纲)手工编写,专门考验 Critic 层在语义邻居上的边界判别
- **tuning/test 隔离**:`random.seed(42)` 分层切分,所有调参在 tuning(21 道),报告数字仅来自 test(52 道);评估脚本默认 `--split test`
- **已知局限**:gold set 由 LLM 生成,可能偏向系统风格;hard-OOS n=6(test)统计精度有限;生产应改为真实用户日志

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
