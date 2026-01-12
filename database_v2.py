"""
先进的RAG向量数据库模块

特性：
1. 分层索引：支持Section/Parent/Child/Summary四种块类型
2. 父子关系检索：检索Child块时可自动获取Parent上下文
3. 多路召回：支持Child检索 + Summary检索混合
4. LLM摘要增强：利用LLM生成章节摘要提升检索效果

使用示例:
    db = HierarchicalVectorDatabase()
    db.process_file("/path/to/document.md", generate_summaries=True)
    results = db.search("your query", top_k=5, return_parent=True)
"""

import pymilvus
from pymilvus import MilvusClient, DataType
from tqdm import tqdm
import json
import os
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import requests
from openai import OpenAI

from chunking import (
    HierarchicalChunker, 
    SummaryGenerator, 
    Chunk, 
    ChunkType,
    chunk_markdown_document
)


@dataclass
class SearchResult:
    """检索结果"""
    chunk_id: str
    chunk_type: str
    text: str
    score: float
    file_name: str
    heading: Optional[str] = None
    parent_text: Optional[str] = None
    section_text: Optional[str] = None
    section_summary: Optional[str] = None


class HierarchicalVectorDatabase:
    """
    分层向量数据库
    
    实现LlamaIndex风格的父子块检索策略：
    - 使用Child块进行精准Embedding检索
    - 命中后自动获取Parent块提供上下文
    - 可选获取Section摘要提供全局视野
    """
    
    # 集合名称
    CHILD_COLLECTION = "rag_children"       # 子块集合（用于检索）
    PARENT_COLLECTION = "rag_parents"       # 父块集合（提供上下文）
    SECTION_COLLECTION = "rag_sections"     # 章节集合
    SUMMARY_COLLECTION = "rag_summaries"    # 摘要集合（可选检索）

    # 添加最大token限制常量
    MAX_EMBEDDING_TOKENS = 8000  # 预留一些余量，设为8000
    MAX_EMBEDDING_CHARS = 24000  # 粗略估计：1 token ≈ 3 字符（中英文混合）
    
    def __init__(
        self,
        database_path: str = "./database_test/hierarchical_rag.db",
        embedding_dim: int = 1024,
        embedding_base_url: str = "http://localhost:7979/v1",
        llm_base_url: str = "http://localhost:28888/v1",
        embedding_model: str = "/data2/home/lijidong/models/bge-m3",
        max_embedding_chars: int = 24000  # 新增参数
    ):
        """
        初始化分层向量数据库
        
        Args:
            database_path: Milvus数据库路径
            embedding_dim: Embedding维度
            embedding_base_url: Embedding模型服务地址
            llm_base_url: LLM服务地址（用于生成摘要）
            embedding_model: Embedding模型路径
            max_embedding_chars: Embedding输入最大字符数（防止超过token限制）
        """
        self.database_path = database_path
        self.embedding_dim = embedding_dim
        self.embedding_base_url = embedding_base_url
        self.llm_base_url = llm_base_url
        self.embedding_model = embedding_model
        self.max_embedding_chars = max_embedding_chars  # 保存配置
        
        # 检查Embedding服务
        if not self._check_embedding_server():
            raise RuntimeError(
                f"请先启动Embedding模型服务，执行如下脚本：\n\n"
                "bash scripts/start_embedding_model.sh\n\n"
            )
        
        # 初始化OpenAI客户端（用于Embedding）
        self.embedding_client = OpenAI(
            base_url=self.embedding_base_url,
            api_key="EMPTY"
        )
        print(f"[INFO] 成功连接到 Embedding 服务 {self.embedding_base_url}")
        
        # 初始化Milvus客户端
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        self.milvus_client = MilvusClient(uri=database_path)
        
        # 初始化所有集合
        self._init_collections()
        
        # 初始化分层切分器
        self.chunker = HierarchicalChunker(
            parent_chunk_size=800,
            child_chunk_size=200,
            parent_overlap=100,
            child_overlap=50
        )
        
        # 尝试初始化摘要生成器
        self.summary_generator = None
        if self._check_llm_server():
            try:
                self.summary_generator = SummaryGenerator(base_url=llm_base_url)
                print(f"[INFO] 摘要生成功能已启用")
            except Exception as e:
                print(f"[WARNING] 摘要生成器初始化失败: {e}")
        else:
            print(f"[INFO] LLM服务未启动，摘要生成功能禁用")
        
        self._print_stats()

    def _truncate_text(self, text: str) -> str:
        """
        截断过长文本以避免超过Embedding模型的token限制
        
        Args:
            text: 原始文本
        
        Returns:
            截断后的文本
        """
        if len(text) <= self.max_embedding_chars:
            return text
        
        # 截断并添加省略标记
        truncated = text[:self.max_embedding_chars]
        
        # 尝试在句子边界截断（避免截断在句子中间）
        # 查找最后一个句号、问号、感叹号或换行符
        for sep in ['\n\n', '\n', '。', '！', '？', '.', '!', '?']:
            last_pos = truncated.rfind(sep)
            if last_pos > self.max_embedding_chars * 0.8:  # 至少保留80%的内容
                truncated = truncated[:last_pos + len(sep)]
                break
        
        return truncated + "\n...[内容已截断]"
    
    def _check_embedding_server(self) -> bool:
        """检查Embedding服务是否可用"""
        try:
            r = requests.get(
                f"{self.embedding_base_url}/models",
                headers={"Authorization": "Bearer EMPTY"},
                timeout=2
            )
            return r.status_code == 200
        except Exception as e:
            print(f"[ERROR] 无法连接到 Embedding 服务: {e}")
            return False
    
    def _check_llm_server(self) -> bool:
        """检查LLM服务是否可用"""
        try:
            r = requests.get(
                f"{self.llm_base_url}/models",
                headers={"Authorization": "Bearer EMPTY"},
                timeout=2
            )
            return r.status_code == 200
        except Exception:
            return False
    
    def _init_collections(self):
        """初始化所有集合"""
        collections = [
            (self.CHILD_COLLECTION, self._create_child_schema),
            (self.PARENT_COLLECTION, self._create_parent_schema),
            (self.SECTION_COLLECTION, self._create_section_schema),
            (self.SUMMARY_COLLECTION, self._create_summary_schema),
        ]
        
        for name, schema_func in collections:
            if not self.milvus_client.has_collection(name):
                schema = schema_func()
                index_params = self.milvus_client.prepare_index_params()
                index_params.add_index(
                    field_name="vector",
                    index_type="FLAT",
                    metric_type="IP"
                )
                self.milvus_client.create_collection(
                    collection_name=name,
                    schema=schema,
                    index_params=index_params,
                    consistency_level="Bounded"
                )
                print(f"[INFO] 创建集合: {name}")
    
    def _create_child_schema(self):
        """Child集合Schema"""
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field(field_name="chunk_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="section_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="position", datatype=DataType.INT64)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        return schema
    
    def _create_parent_schema(self):
        """Parent集合Schema"""
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field(field_name="chunk_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="section_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="heading", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="child_ids", datatype=DataType.VARCHAR, max_length=65535)  # JSON序列化的列表
        schema.add_field(field_name="position", datatype=DataType.INT64)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        return schema
    
    def _create_section_schema(self):
        """Section集合Schema"""
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field(field_name="chunk_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="heading", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="heading_level", datatype=DataType.INT64)
        schema.add_field(field_name="position", datatype=DataType.INT64)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        return schema
    
    def _create_summary_schema(self):
        """Summary集合Schema"""
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field(field_name="chunk_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="section_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="heading", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="position", datatype=DataType.INT64)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        return schema
    
    def _print_stats(self):
        """打印数据库统计信息"""
        print("\n[INFO] 数据库统计:")
        for collection in [self.CHILD_COLLECTION, self.PARENT_COLLECTION, 
                          self.SECTION_COLLECTION, self.SUMMARY_COLLECTION]:
            if self.milvus_client.has_collection(collection):
                stats = self.milvus_client.get_collection_stats(collection)
                count = stats.get("row_count", 0)
                print(f"  - {collection}: {count} 条记录")
    
    def embedding(self, text: str) -> Optional[List[float]]:
        """生成文本的Embedding向量"""
        try:
            # 截断过长文本
            text = self._truncate_text(text)
            
            response = self.embedding_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[ERROR] Embedding生成失败: {e}")
            return None
    
    def batch_embedding(self, texts: List[str], batch_size: int = 32) -> List[Optional[List[float]]]:
        """批量生成Embedding"""
        embeddings = []
        
        # 预处理：截断所有过长文本
        truncated_texts = [self._truncate_text(text) for text in texts]
        
        # 记录被截断的文本数量
        truncated_count = sum(1 for orig, trunc in zip(texts, truncated_texts) 
                              if len(orig) != len(trunc))
        if truncated_count > 0:
            print(f"[INFO] {truncated_count} 个文本因超过长度限制被截断")
        
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i:i + batch_size]
            try:
                response = self.embedding_client.embeddings.create(
                    input=batch,
                    model=self.embedding_model
                )
                for item in response.data:
                    embeddings.append(item.embedding)
            except Exception as e:
                print(f"[ERROR] 批量Embedding失败: {e}")
                # 如果批量失败，尝试逐个处理
                print(f"[INFO] 尝试逐个处理这批文本...")
                for text in batch:
                    single_embedding = self._single_embedding_with_retry(text)
                    embeddings.append(single_embedding)
        
        return embeddings
    
    def _single_embedding_with_retry(self, text: str, max_retries: int = 2) -> Optional[List[float]]:
        """
        单个文本Embedding，带重试和进一步截断
        
        Args:
            text: 文本
            max_retries: 最大重试次数
        
        Returns:
            Embedding向量或None
        """
        current_text = text
        
        for attempt in range(max_retries + 1):
            try:
                response = self.embedding_client.embeddings.create(
                    input=current_text,
                    model=self.embedding_model
                )
                return response.data[0].embedding
            except Exception as e:
                if "maximum context length" in str(e) and attempt < max_retries:
                    # 进一步截断（每次减少30%）
                    new_length = int(len(current_text) * 0.7)
                    current_text = current_text[:new_length] + "\n...[内容已截断]"
                    print(f"[WARNING] 文本仍然过长，进一步截断到 {new_length} 字符")
                else:
                    print(f"[ERROR] 单个Embedding失败: {e}")
                    return None
        
        return None
    
    def process_file(
        self,
        file_path: str,
        generate_summaries: bool = False,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        处理单个Markdown文件
        
        Args:
            file_path: 文件路径
            generate_summaries: 是否生成摘要
            show_progress: 是否显示进度
        
        Returns:
            各类型块的数量统计
        """
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        
        file_name = os.path.basename(file_path)
        
        # 分层切分
        chunks = self.chunker.chunk_document(markdown_text, file_name)
        
        stats = {
            "sections": 0,
            "parents": 0,
            "children": 0,
            "summaries": 0
        }
        
        # 处理Sections
        if chunks['sections']:
            section_data = self._prepare_section_data(chunks['sections'], show_progress)
            if section_data:
                self.milvus_client.insert(self.SECTION_COLLECTION, section_data)
                stats['sections'] = len(section_data)
        
        # 处理Parents
        if chunks['parents']:
            parent_data = self._prepare_parent_data(chunks['parents'], show_progress)
            if parent_data:
                self.milvus_client.insert(self.PARENT_COLLECTION, parent_data)
                stats['parents'] = len(parent_data)
        
        # 处理Children
        if chunks['children']:
            child_data = self._prepare_child_data(chunks['children'], show_progress)
            if child_data:
                self.milvus_client.insert(self.CHILD_COLLECTION, child_data)
                stats['children'] = len(child_data)
        
        # 生成并处理Summaries
        if generate_summaries and self.summary_generator:
            summaries = []
            iterator = tqdm(chunks['sections'], desc="生成摘要") if show_progress else chunks['sections']
            for section in iterator:
                summary = self.summary_generator.generate_section_summary(section)
                if summary:
                    summaries.append(summary)
            
            if summaries:
                summary_data = self._prepare_summary_data(summaries, show_progress)
                if summary_data:
                    self.milvus_client.insert(self.SUMMARY_COLLECTION, summary_data)
                    stats['summaries'] = len(summary_data)
        
        print(f"[INFO] 文件处理完成: {file_name}")
        print(f"  - Sections: {stats['sections']}, Parents: {stats['parents']}, "
              f"Children: {stats['children']}, Summaries: {stats['summaries']}")
        
        return stats
    
    def _prepare_child_data(self, chunks: List[Chunk], show_progress: bool) -> List[dict]:
        """准备Child数据"""
        texts = [c.text for c in chunks]
        embeddings = self.batch_embedding(texts)
        
        data = []
        iterator = zip(chunks, embeddings)
        if show_progress:
            iterator = tqdm(list(iterator), desc="处理Child块")
        
        for chunk, embedding in iterator:
            if embedding is None:
                continue
            data.append({
                "id": chunk.id,
                "chunk_type": chunk.chunk_type.value,
                "text": chunk.text,
                "file_name": chunk.file_name,
                "parent_id": chunk.parent_id or "",
                "section_id": chunk.section_id or "",
                "position": chunk.position,
                "vector": embedding
            })
        return data
    
    def _prepare_parent_data(self, chunks: List[Chunk], show_progress: bool) -> List[dict]:
        """准备Parent数据"""
        texts = [c.text for c in chunks]
        embeddings = self.batch_embedding(texts)
        
        data = []
        iterator = zip(chunks, embeddings)
        if show_progress:
            iterator = tqdm(list(iterator), desc="处理Parent块")
        
        for chunk, embedding in iterator:
            if embedding is None:
                continue
            data.append({
                "id": chunk.id,
                "chunk_type": chunk.chunk_type.value,
                "text": chunk.text,
                "file_name": chunk.file_name,
                "section_id": chunk.section_id or "",
                "heading": chunk.heading or "",
                "child_ids": json.dumps(chunk.child_ids),
                "position": chunk.position,
                "vector": embedding
            })
        return data
    
    def _prepare_section_data(self, chunks: List[Chunk], show_progress: bool) -> List[dict]:
        """准备Section数据"""
        texts = [c.text for c in chunks]
        embeddings = self.batch_embedding(texts)
        
        data = []
        iterator = zip(chunks, embeddings)
        if show_progress:
            iterator = tqdm(list(iterator), desc="处理Section块")
        
        for chunk, embedding in iterator:
            if embedding is None:
                continue
            data.append({
                "id": chunk.id,
                "chunk_type": chunk.chunk_type.value,
                "text": chunk.text,
                "file_name": chunk.file_name,
                "heading": chunk.heading or "",
                "heading_level": chunk.heading_level,
                "position": chunk.position,
                "vector": embedding
            })
        return data
    
    def _prepare_summary_data(self, chunks: List[Chunk], show_progress: bool) -> List[dict]:
        """准备Summary数据"""
        texts = [c.text for c in chunks]
        embeddings = self.batch_embedding(texts)
        
        data = []
        iterator = zip(chunks, embeddings)
        if show_progress:
            iterator = tqdm(list(iterator), desc="处理Summary块")
        
        for chunk, embedding in iterator:
            if embedding is None:
                continue
            data.append({
                "id": chunk.id,
                "chunk_type": chunk.chunk_type.value,
                "text": chunk.text,
                "file_name": chunk.file_name,
                "section_id": chunk.section_id or "",
                "heading": chunk.heading or "",
                "position": chunk.position,
                "vector": embedding
            })
        return data
    
    def process_files(
        self,
        file_list: List[str],
        generate_summaries: bool = False
    ) -> Dict[str, int]:
        """批量处理文件"""
        total_stats = {"sections": 0, "parents": 0, "children": 0, "summaries": 0}
        
        for file_path in tqdm(file_list, desc="处理文档"):
            if not os.path.exists(file_path):
                print(f"[WARNING] 文件不存在: {file_path}")
                continue
            
            stats = self.process_file(file_path, generate_summaries, show_progress=False)
            for key in total_stats:
                total_stats[key] += stats[key]
        
        print(f"\n[INFO] 批量处理完成，共处理 {len(file_list)} 个文件")
        print(f"  总计: Sections={total_stats['sections']}, Parents={total_stats['parents']}, "
              f"Children={total_stats['children']}, Summaries={total_stats['summaries']}")
        
        return total_stats
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        return_parent: bool = True,
        return_section_summary: bool = False,
        use_summary_search: bool = False,
        summary_weight: float = 0.3
    ) -> List[SearchResult]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            return_parent: 是否返回父块上下文
            return_section_summary: 是否返回章节摘要
            use_summary_search: 是否同时搜索摘要（多路召回）
            summary_weight: 摘要搜索的权重（仅当use_summary_search=True时有效）
        
        Returns:
            SearchResult列表
        """
        query_embedding = self.embedding(query)
        if query_embedding is None:
            return []
        
        results = []
        
        # 主要检索：Child块
        child_results = self.milvus_client.search(
            collection_name=self.CHILD_COLLECTION,
            data=[query_embedding],
            limit=top_k * 2 if use_summary_search else top_k,
            output_fields=["id", "text", "file_name", "parent_id", "section_id", "chunk_type"]
        )
        
        # 处理Child结果
        seen_parents = set()
        for hit in child_results[0]:
            entity = hit.get("entity", hit)
            chunk_id = entity.get("id", "")
            parent_id = entity.get("parent_id", "")
            section_id = entity.get("section_id", "")
            
            result = SearchResult(
                chunk_id=chunk_id,
                chunk_type=entity.get("chunk_type", "child"),
                text=entity.get("text", ""),
                score=hit.get("distance", 0),
                file_name=entity.get("file_name", ""),
            )
            
            # 获取Parent上下文
            if return_parent and parent_id and parent_id not in seen_parents:
                parent_text = self._get_parent_text(parent_id)
                result.parent_text = parent_text
                seen_parents.add(parent_id)
            
            # 获取Section摘要
            if return_section_summary and section_id:
                summary_text = self._get_section_summary(section_id)
                result.section_summary = summary_text
            
            results.append(result)
        
        # 可选：Summary多路召回
        if use_summary_search:
            summary_results = self.milvus_client.search(
                collection_name=self.SUMMARY_COLLECTION,
                data=[query_embedding],
                limit=top_k // 2,
                output_fields=["id", "text", "file_name", "section_id", "heading", "chunk_type"]
            )
            
            for hit in summary_results[0]:
                entity = hit.get("entity", hit)
                result = SearchResult(
                    chunk_id=entity.get("id", ""),
                    chunk_type="summary",
                    text=entity.get("text", ""),
                    score=hit.get("distance", 0) * summary_weight,
                    file_name=entity.get("file_name", ""),
                    heading=entity.get("heading", "")
                )
                results.append(result)
        
        # 按分数排序并返回top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _get_parent_text(self, parent_id: str) -> Optional[str]:
        """获取Parent块文本"""
        try:
            results = self.milvus_client.query(
                collection_name=self.PARENT_COLLECTION,
                filter=f'id == "{parent_id}"',
                output_fields=["text", "heading"]
            )
            if results:
                heading = results[0].get("heading", "")
                text = results[0].get("text", "")
                if heading:
                    return f"[{heading}]\n{text}"
                return text
        except Exception as e:
            print(f"[WARNING] 获取Parent失败: {e}")
        return None
    
    def _get_section_summary(self, section_id: str) -> Optional[str]:
        """获取Section摘要"""
        try:
            results = self.milvus_client.query(
                collection_name=self.SUMMARY_COLLECTION,
                filter=f'section_id == "{section_id}"',
                output_fields=["text"]
            )
            if results:
                return results[0].get("text")
        except Exception as e:
            pass
        return None
    
    def search_with_context(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        搜索并返回完整上下文（便于直接给LLM使用）
        
        Args:
            query: 查询文本
            top_k: 返回数量
        
        Returns:
            包含chunk和上下文的字典列表
        """
        results = self.search(
            query,
            top_k=top_k,
            return_parent=True,
            return_section_summary=True
        )
        
        context_list = []
        for r in results:
            context = {
                "matched_chunk": r.text,
                "score": r.score,
                "file_name": r.file_name,
            }
            
            # 优先使用Parent作为上下文
            if r.parent_text:
                context["context"] = r.parent_text
            else:
                context["context"] = r.text
            
            # 添加摘要作为背景
            if r.section_summary:
                context["section_summary"] = r.section_summary
            
            context_list.append(context)
        
        return context_list
    
    def get_retrieval_context(self, query: str, top_k: int = 3) -> str:
        """
        获取检索上下文（格式化后的文本，可直接用于RAG）
        
        Args:
            query: 查询文本
            top_k: 返回数量
        
        Returns:
            格式化的上下文文本
        """
        results = self.search_with_context(query, top_k)
        
        context_parts = []
        for i, r in enumerate(results, 1):
            part = f"### 参考文档 {i} (来源: {r['file_name']}, 相关度: {r['score']:.3f})\n"
            
            if r.get("section_summary"):
                part += f"**摘要**: {r['section_summary']}\n\n"
            
            part += f"**内容**:\n{r['context']}\n"
            context_parts.append(part)
        
        return "\n---\n".join(context_parts)
    
    def clear_all(self):
        """清空所有集合"""
        for collection in [self.CHILD_COLLECTION, self.PARENT_COLLECTION,
                          self.SECTION_COLLECTION, self.SUMMARY_COLLECTION]:
            if self.milvus_client.has_collection(collection):
                self.milvus_client.drop_collection(collection)
        
        self._init_collections()
        print("[INFO] 所有集合已清空")


# 兼容旧接口
class VectorDatabase(HierarchicalVectorDatabase):
    """
    向后兼容的VectorDatabase类
    
    继承自HierarchicalVectorDatabase，提供与旧代码兼容的接口
    """
    
    def __init__(self, collection_name="chunk", embedding_dim=1024):
        # 调用新类的初始化
        super().__init__(
            database_path="./database/OIR.db",
            embedding_dim=embedding_dim
        )
        self.collect_name = collection_name
        self.id_now = 0
    
    def process_a_file(self, file_path: str):
        """兼容旧接口的文件处理方法"""
        return self.process_file(file_path, generate_summaries=False)
    
    def search_embedding(self, query_text: str, top_k: int = 3) -> List[str]:
        """兼容旧接口的搜索方法，返回文本列表"""
        results = self.search(query_text, top_k=top_k, return_parent=True)
        
        # 返回Parent上下文（如果有）或Child文本
        texts = []
        for r in results:
            if r.parent_text:
                texts.append(r.parent_text)
            else:
                texts.append(r.text)
        
        return texts


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分层RAG向量数据库")
    parser.add_argument("--mode", choices=["insert", "search", "demo"], default="demo",
                       help="运行模式: insert=插入数据, search=搜索, demo=演示")
    parser.add_argument("--path", type=str, help="要处理的文件或目录路径")
    parser.add_argument("--query", type=str, help="搜索查询")
    parser.add_argument("--summaries", action="store_true", help="是否生成摘要")
    parser.add_argument("--top-k", type=int, default=3, help="返回结果数量")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        # 演示模式
        print("=" * 60)
        print("分层RAG向量数据库演示")
        print("=" * 60)
        
        db = HierarchicalVectorDatabase()
        
        # 示例查询
        query = "Citing Foreign Affairs Records"
        print(f"\n查询: {query}\n")
        
        results = db.search(query, top_k=3, return_parent=True)
        for i, r in enumerate(results, 1):
            print(f"\n--- 结果 {i} (分数: {r.score:.3f}) ---")
            print(f"匹配块: {r.text[:200]}...")
            if r.parent_text:
                print(f"\n上下文: {r.parent_text[:300]}...")
    
    elif args.mode == "insert":
        if not args.path:
            print("请提供 --path 参数")
            exit(1)
        
        db = HierarchicalVectorDatabase()
        
        if os.path.isfile(args.path):
            db.process_file(args.path, generate_summaries=args.summaries)
        elif os.path.isdir(args.path):
            # 查找所有.md文件
            md_files = []
            for root, dirs, files in os.walk(args.path):
                for f in files:
                    if f.endswith('.md'):
                        md_files.append(os.path.join(root, f))
            print(f"找到 {len(md_files)} 个Markdown文件")
            db.process_files(md_files, generate_summaries=args.summaries)
    
    elif args.mode == "search":
        if not args.query:
            print("请提供 --query 参数")
            exit(1)
        
        db = HierarchicalVectorDatabase()
        # context = db.get_retrieval_context(args.query, top_k=args.top_k)
        context = db.search_with_context(args.query, top_k=args.top_k)
        print(context)
