"""
分层RAG系统使用示例

演示如何使用新的分层切分和检索系统
"""

import os
import json
from database_v2 import HierarchicalVectorDatabase


def example_1_basic_usage():
    """示例1: 基础用法 - 处理文件和搜索"""
    print("\n" + "=" * 60)
    print("示例1: 基础用法")
    print("=" * 60)
    
    # 初始化数据库
    db = HierarchicalVectorDatabase(
        database_path="./database/example_rag.db",
        embedding_base_url="http://localhost:7979/v1",
        llm_base_url="http://localhost:8888/v1"
    )
    
    # 处理单个文件
    file_path = "/data2/home/lijidong/vllm-qwen/datafiles/guowu_data/America/downloadsnew/taft/epub/frus1907p2.md"
    
    if os.path.exists(file_path):
        stats = db.process_file(file_path, generate_summaries=False)
        print(f"处理结果: {stats}")
    else:
        print(f"文件不存在: {file_path}")
        return
    
    # 搜索
    query = "treaty between Japan and Russia"
    results = db.search(query, top_k=3, return_parent=True)
    
    print(f"\n查询: {query}")
    print("-" * 40)
    
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i} (分数: {result.score:.3f})")
        print(f"文件: {result.file_name}")
        print(f"匹配内容: {result.text[:200]}...")
        if result.parent_text:
            print(f"上下文: {result.parent_text[:300]}...")


def example_2_with_summaries():
    """示例2: 使用LLM摘要增强检索"""
    print("\n" + "=" * 60)
    print("示例2: 使用LLM摘要增强")
    print("=" * 60)
    
    db = HierarchicalVectorDatabase(
        database_path="./database/example_rag_summary.db"
    )
    
    # 处理文件并生成摘要
    file_path = "/data2/home/lijidong/vllm-qwen/datafiles/guowu_data/America/downloadsnew/taft/epub/frus1908.md"
    
    if os.path.exists(file_path):
        # 启用摘要生成
        stats = db.process_file(file_path, generate_summaries=True)
        print(f"处理结果: {stats}")
    else:
        print(f"文件不存在: {file_path}")
        return
    
    # 使用多路召回搜索（同时搜索Child和Summary）
    query = "diplomatic relations with China"
    results = db.search(
        query, 
        top_k=5, 
        return_parent=True,
        return_section_summary=True,
        use_summary_search=True
    )
    
    print(f"\n查询: {query}")
    for i, result in enumerate(results, 1):
        print(f"\n--- 结果 {i} (分数: {result.score:.3f}, 类型: {result.chunk_type}) ---")
        if result.section_summary:
            print(f"章节摘要: {result.section_summary}")
        print(f"内容: {result.text[:200]}...")


def example_3_batch_processing():
    """示例3: 批量处理文件"""
    print("\n" + "=" * 60)
    print("示例3: 批量处理文件")
    print("=" * 60)
    
    db = HierarchicalVectorDatabase(
        database_path="./database/batch_rag.db"
    )
    
    # 查找目录下的所有md文件
    root_path = "/data2/home/lijidong/vllm-qwen/datafiles/guowu_data/America/downloadsnew/taft/epub/"
    
    if os.path.exists(root_path):
        md_files = [
            os.path.join(root_path, f) 
            for f in os.listdir(root_path) 
            if f.endswith('.md')
        ][:3]  # 只处理前3个作为示例
        
        print(f"找到 {len(md_files)} 个文件")
        stats = db.process_files(md_files, generate_summaries=False)
        print(f"总计: {stats}")
    else:
        print(f"目录不存在: {root_path}")


def example_4_rag_context():
    """示例4: 获取RAG上下文用于LLM"""
    print("\n" + "=" * 60)
    print("示例4: 获取RAG上下文")
    print("=" * 60)
    
    db = HierarchicalVectorDatabase(
        database_path="./database/batch_rag.db"
    )
    
    query = "peace conference negotiations"
    
    # 获取格式化的上下文（可直接用于LLM prompt）
    context = db.get_retrieval_context(query, top_k=3)
    
    print(f"查询: {query}")
    print("-" * 40)
    print(context)
    
    # 也可以获取结构化的数据
    print("\n\n结构化结果:")
    results = db.search_with_context(query, top_k=3)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str)[:1000])


def example_5_compatible_interface():
    """示例5: 兼容旧接口"""
    print("\n" + "=" * 60)
    print("示例5: 兼容旧代码接口")
    print("=" * 60)
    
    # 使用VectorDatabase类保持与旧代码兼容
    from database_v2 import VectorDatabase
    
    db = VectorDatabase(collection_name="chunk")
    
    # 旧接口: search_embedding 返回文本列表
    query = "foreign affairs"
    results = db.search_embedding(query, top_k=3)
    
    print(f"查询: {query}")
    for i, text in enumerate(results, 1):
        print(f"\n结果 {i}: {text[:200]}...")


def example_6_clear_and_rebuild():
    """示例6: 清空数据库并重建"""
    print("\n" + "=" * 60)
    print("示例6: 清空并重建数据库")
    print("=" * 60)
    
    db = HierarchicalVectorDatabase(
        database_path="./database/rebuild_test.db"
    )
    
    # 清空所有数据
    db.clear_all()
    print("数据库已清空")
    
    # 重新导入数据
    # db.process_file("your_file.md", generate_summaries=True)


if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("基础用法", example_1_basic_usage),
        "2": ("LLM摘要增强", example_2_with_summaries),
        "3": ("批量处理", example_3_batch_processing),
        "4": ("RAG上下文", example_4_rag_context),
        "5": ("兼容旧接口", example_5_compatible_interface),
        "6": ("清空重建", example_6_clear_and_rebuild),
    }
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            examples[choice][1]()
        else:
            print(f"未知示例: {choice}")
    else:
        print("分层RAG系统使用示例")
        print("-" * 40)
        print("可用示例:")
        for k, (name, _) in examples.items():
            print(f"  {k}: {name}")
        print("\n运行方式: python example_usage.py <编号>")
        print("例如: python example_usage.py 1")
