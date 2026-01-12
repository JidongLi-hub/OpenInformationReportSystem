"""
完整的RAG服务

整合：
1. 分层文档切分和索引
2. 智能检索（Parent-Child策略）
3. LLM生成回答

使用方法:
    # 启动服务
    python rag_service.py --mode serve --port 8080
    
    # 命令行问答
    python rag_service.py --mode chat
    
    # 处理文档
    python rag_service.py --mode index --path /path/to/docs
"""

import os
import json
import argparse
from typing import List, Dict, Optional
from dataclasses import dataclass
from openai import OpenAI

from database_v2 import HierarchicalVectorDatabase


@dataclass
class RAGConfig:
    """RAG配置"""
    # 数据库配置
    database_path: str = "./database/hierarchical_rag.db"
    
    # Embedding服务
    embedding_base_url: str = "http://localhost:7979/v1"
    embedding_model: str = "/data2/home/lijidong/models/bge-m3"
    
    # LLM服务
    llm_base_url: str = "http://localhost:8888/v1"
    llm_model: Optional[str] = None  # 自动检测
    
    # 检索配置
    top_k: int = 5
    return_parent: bool = True
    use_summary_search: bool = False
    
    # 生成配置
    max_tokens: int = 2048
    temperature: float = 0.7
    
    # 系统提示词
    system_prompt: str = """你是一个专业的历史文献分析助手。基于提供的参考资料回答用户问题。

请遵循以下原则：
1. 优先使用参考资料中的信息
2. 如果资料不足以回答问题，请明确说明
3. 引用来源时注明文档名称
4. 保持回答准确、客观"""


class RAGService:
    """
    完整的RAG服务
    
    提供文档索引、检索和问答功能
    """
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
        # 初始化向量数据库
        self.vector_db = HierarchicalVectorDatabase(
            database_path=self.config.database_path,
            embedding_base_url=self.config.embedding_base_url,
            llm_base_url=self.config.llm_base_url,
            embedding_model=self.config.embedding_model
        )
        
        # 初始化LLM客户端
        self.llm_client = OpenAI(
            base_url=self.config.llm_base_url,
            api_key="EMPTY"
        )
        
        # 获取模型名称
        if not self.config.llm_model:
            self._detect_llm_model()
    
    def _detect_llm_model(self):
        """自动检测LLM模型"""
        try:
            models = self.llm_client.models.list()
            if models.data:
                self.config.llm_model = models.data[0].id
                print(f"[INFO] 检测到LLM模型: {self.config.llm_model}")
        except Exception as e:
            print(f"[WARNING] 无法检测LLM模型: {e}")
            self.config.llm_model = "default"
    
    def index_file(self, file_path: str, generate_summaries: bool = False) -> Dict:
        """索引单个文件"""
        return self.vector_db.process_file(file_path, generate_summaries)
    
    def index_directory(self, dir_path: str, generate_summaries: bool = False) -> Dict:
        """索引目录下的所有Markdown文件"""
        md_files = []
        for root, dirs, files in os.walk(dir_path):
            for f in files:
                if f.endswith('.md'):
                    md_files.append(os.path.join(root, f))
        
        print(f"[INFO] 找到 {len(md_files)} 个Markdown文件")
        return self.vector_db.process_files(md_files, generate_summaries)
    
    def retrieve(self, query: str) -> List[Dict]:
        """检索相关文档"""
        return self.vector_db.search_with_context(
            query,
            top_k=self.config.top_k
        )
    
    def get_context(self, query: str) -> str:
        """获取格式化的检索上下文"""
        return self.vector_db.get_retrieval_context(
            query,
            top_k=self.config.top_k
        )
    
    def generate(self, query: str, context: str) -> str:
        """使用LLM生成回答"""
        prompt = f"""参考资料：
{context}

用户问题：{query}

请基于以上参考资料回答问题。如果资料中没有相关信息，请如实说明。"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[ERROR] 生成回答失败: {e}"
    
    def ask(self, query: str, show_context: bool = False) -> Dict:
        """
        完整的RAG问答流程
        
        Args:
            query: 用户问题
            show_context: 是否返回检索上下文
        
        Returns:
            包含answer和可选context的字典
        """
        # 检索
        context = self.get_context(query)
        
        # 生成
        answer = self.generate(query, context)
        
        result = {"query": query, "answer": answer}
        if show_context:
            result["context"] = context
        
        return result
    
    def chat(self):
        """交互式问答"""
        print("\n" + "=" * 60)
        print("RAG 交互式问答")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'context' 显示/隐藏检索上下文")
        print("=" * 60 + "\n")
        
        show_context = False
        
        while True:
            try:
                query = input("\n你: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n再见！")
                break
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            if query.lower() == 'context':
                show_context = not show_context
                print(f"[INFO] 上下文显示: {'开启' if show_context else '关闭'}")
                continue
            
            print("\n[正在检索和生成...]")
            result = self.ask(query, show_context)
            
            if show_context and result.get("context"):
                print("\n📚 检索上下文:")
                print("-" * 40)
                print(result["context"][:1500] + "..." if len(result.get("context", "")) > 1500 else result.get("context", ""))
                print("-" * 40)
            
            print(f"\n🤖 助手: {result['answer']}")


def run_flask_server(rag_service: RAGService, port: int = 8080):
    """运行Flask API服务器"""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("[ERROR] 需要安装flask: pip install flask")
        return
    
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok"})
    
    @app.route('/ask', methods=['POST'])
    def ask():
        data = request.json
        query = data.get('query', '')
        show_context = data.get('show_context', False)
        
        if not query:
            return jsonify({"error": "query is required"}), 400
        
        result = rag_service.ask(query, show_context)
        return jsonify(result)
    
    @app.route('/retrieve', methods=['POST'])
    def retrieve():
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "query is required"}), 400
        
        results = rag_service.retrieve(query)
        return jsonify({"results": results})
    
    @app.route('/index', methods=['POST'])
    def index():
        data = request.json
        path = data.get('path', '')
        generate_summaries = data.get('generate_summaries', False)
        
        if not path:
            return jsonify({"error": "path is required"}), 400
        
        if os.path.isfile(path):
            stats = rag_service.index_file(path, generate_summaries)
        elif os.path.isdir(path):
            stats = rag_service.index_directory(path, generate_summaries)
        else:
            return jsonify({"error": "path not found"}), 404
        
        return jsonify({"stats": stats})
    
    print(f"\n[INFO] 启动RAG API服务器: http://localhost:{port}")
    print("API端点:")
    print("  POST /ask - 问答 (body: {query, show_context})")
    print("  POST /retrieve - 检索 (body: {query})")
    print("  POST /index - 索引文档 (body: {path, generate_summaries})")
    print("  GET /health - 健康检查")
    
    app.run(host='0.0.0.0', port=port)


def main():
    parser = argparse.ArgumentParser(description="RAG服务")
    parser.add_argument("--mode", choices=["chat", "serve", "index", "ask"],
                       default="chat", help="运行模式")
    parser.add_argument("--path", type=str, help="文档路径（用于index模式）")
    parser.add_argument("--query", type=str, help="查询（用于ask模式）")
    parser.add_argument("--port", type=int, default=8080, help="API端口")
    parser.add_argument("--summaries", action="store_true", help="生成摘要")
    parser.add_argument("--db", type=str, default="./database/hierarchical_rag.db",
                       help="数据库路径")
    
    args = parser.parse_args()
    
    # 初始化配置
    config = RAGConfig(database_path=args.db)
    rag = RAGService(config)
    
    if args.mode == "chat":
        rag.chat()
    
    elif args.mode == "serve":
        run_flask_server(rag, args.port)
    
    elif args.mode == "index":
        if not args.path:
            print("[ERROR] 请提供 --path 参数")
            return
        
        if os.path.isfile(args.path):
            rag.index_file(args.path, args.summaries)
        elif os.path.isdir(args.path):
            rag.index_directory(args.path, args.summaries)
        else:
            print(f"[ERROR] 路径不存在: {args.path}")
    
    elif args.mode == "ask":
        if not args.query:
            print("[ERROR] 请提供 --query 参数")
            return
        
        result = rag.ask(args.query, show_context=True)
        print(f"\n问题: {result['query']}")
        print(f"\n回答: {result['answer']}")
        if result.get('context'):
            print(f"\n参考资料:\n{result['context']}")


if __name__ == "__main__":
    main()
