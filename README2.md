# 开源情报项目

## 项目架构

```
vllm-qwen/
├── chunking.py          # 📦 分层文档切分模块 (新)
├── database_v2.py       # 🗄️ 分层向量数据库 (新)
├── rag_service.py       # 🤖 完整RAG服务 (新)
├── example_usage.py     # 📚 使用示例 (新)
├── database.py          # 旧版数据库（保留兼容）
├── format.py            # 文档格式转换
├── server.py            # Web服务
└── ...
```

---

## 🚀 新版分层RAG系统

### 核心特性

采用 **LlamaIndex 风格的父子块切分策略**，显著提升检索精度和上下文完整性：

| 层级 | 用途 | 大小 |
|------|------|------|
| **Section** | 章节，保留完整语义单元 | 无限制 |
| **Parent** | 父块，给LLM阅读的上下文 | ~800词 |
| **Child** | 子块，用于Embedding检索 | ~200词 |
| **Summary** | LLM摘要，增强语义检索 | ~50词 |

**工作原理：**
1. 使用 **Child块** 进行精准向量检索
2. 命中后，通过 `parent_id` 索引获取 **Parent块** 提供完整上下文
3. 可选通过 `section_id` 获取 **章节摘要** 提供全局视野

### 快速开始

#### 1. 启动服务

```bash
# 启动Embedding模型 (端口7979)
bash scripts/start_embedding_model.sh

# 启动LLM模型 (端口8888) - 可选，用于生成摘要
python start_vllm.py --port 8888
```

#### 2. 索引文档

```bash
# 索引单个文件
python database_v2.py --mode insert --path /path/to/document.md

# 索引目录（所有.md文件）
python database_v2.py --mode insert --path /path/to/docs/

# 启用LLM摘要增强
python database_v2.py --mode insert --path /path/to/docs/ --summaries
```

#### 3. 搜索测试

```bash
# 命令行搜索
python database_v2.py --mode search --query "treaty negotiations" --top-k 5

# 交互式问答
python rag_service.py --mode chat

# 启动API服务
python rag_service.py --mode serve --port 8080
```

### Python API 使用

```python
from database_v2 import HierarchicalVectorDatabase

# 初始化
db = HierarchicalVectorDatabase(
    database_path="./database/my_rag.db",
    embedding_base_url="http://localhost:7979/v1",
    llm_base_url="http://localhost:8888/v1"
)

# 索引文档
db.process_file("/path/to/doc.md", generate_summaries=True)

# 搜索（自动返回Parent上下文）
results = db.search("your query", top_k=5, return_parent=True)

for r in results:
    print(f"Score: {r.score:.3f}")
    print(f"Matched: {r.text[:100]}...")
    print(f"Context: {r.parent_text[:200]}...")  # 更完整的上下文

# 获取RAG上下文（格式化后可直接给LLM）
context = db.get_retrieval_context("your query", top_k=3)
print(context)
```

### 完整RAG问答

```python
from rag_service import RAGService, RAGConfig

# 自定义配置
config = RAGConfig(
    database_path="./database/my_rag.db",
    top_k=5,
    temperature=0.7,
    system_prompt="你是一个专业的历史文献分析助手..."
)

rag = RAGService(config)

# 问答
result = rag.ask("What were the main diplomatic issues in 1907?")
print(result['answer'])
```

### API 端点 (serve模式)

```bash
python rag_service.py --mode serve --port 8080
```

| 端点 | 方法 | 说明 |
|------|------|------|
| `/ask` | POST | 问答 `{query, show_context}` |
| `/retrieve` | POST | 检索 `{query}` |
| `/index` | POST | 索引 `{path, generate_summaries}` |
| `/health` | GET | 健康检查 |

---

## 数据处理

### 1. 格式转换

爬取的原始文档，大多为epub(电子书)或者pdf格式，难以直接用于分块处理和向量化。

```python
from format import convert_epub_files_to_markdown, convert_pdf_to_markdown

# EPUB转Markdown
convert_epub_files_to_markdown("/path/to/epubs/")

# PDF转Markdown (使用marker)
convert_pdf_to_markdown("/path/to/pdfs/")
```

### 2. 分块处理

新版系统自动进行智能分块：

```python
from chunking import chunk_markdown_document

with open("document.md") as f:
    text = f.read()

result = chunk_markdown_document(
    text,
    file_name="document.md",
    parent_chunk_size=800,    # 父块大小
    child_chunk_size=200,     # 子块大小
    generate_summaries=True   # 启用摘要
)

print(f"Sections: {len(result['sections'])}")
print(f"Parents: {len(result['parents'])}")
print(f"Children: {len(result['children'])}")
print(f"Summaries: {len(result['summaries'])}")
```

---

## 智能分析与态势感知系统部署

本模块负责将处理好的数据与大模型结合（RAG），生成智能态势报告，并通过可视化界面呈现。

### 1. 环境准备

```bash
# 基础服务依赖
pip install fastapi uvicorn requests python-multipart

# 向量数据库与AI接口依赖
pip install pymilvus openai

# 大模型推理依赖
pip install vllm

# 可选：API服务
pip install flask
```

### 2. 启动大模型推理服务 (vLLM)

```bash
# Embedding模型 (必需)
bash scripts/start_embedding_model.sh

# LLM模型 (用于问答和摘要)
python start_vllm.py
```

### 3. 启动应用服务

```bash
# 新版RAG服务 (推荐)
python rag_service.py --mode serve --port 8080

# 或使用原有的Web界面
python server.py
```

### 4. 浏览器访问

SSH 端口映射:
```bash
ssh -L 8080:localhost:8080 用户名@服务器IP
```

访问: http://localhost:8080

---

## 备选方案：DeepSeek 云端 API 模式

当本地服务器显卡资源紧张时，可使用 DeepSeek-V3 云端大模型。

### 1. 配置密钥

创建 `.env` 文件:
```properties
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 2. 启动服务

```bash
python server_deepseek_api.py
```

---

## 兼容旧代码

如果你有使用旧版 `VectorDatabase` 的代码，可以无缝迁移：

```python
# 旧代码
from database import VectorDatabase
db = VectorDatabase()
results = db.search_embedding("query", top_k=3)

# 新代码 (完全兼容)
from database_v2 import VectorDatabase
db = VectorDatabase()
results = db.search_embedding("query", top_k=3)  # 自动返回Parent上下文
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `chunking.py` | 分层切分模块，包含MarkdownParser、HierarchicalChunker、SummaryGenerator |
| `database_v2.py` | 分层向量数据库，支持Section/Parent/Child/Summary四层索引 |
| `rag_service.py` | 完整RAG服务，整合检索和LLM生成 |
| `example_usage.py` | 使用示例代码 |
| `database.py` | 旧版数据库（保留兼容） |
| `format.py` | 文档格式转换工具 |
