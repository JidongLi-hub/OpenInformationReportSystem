from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from typing import List
import re
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pypandoc
from ebooklib import epub
from bs4 import BeautifulSoup
import html2text


def convert_epub_to_markdown(input_epub_path: str, output_md_path: str):
    """从 EPUB 提取内容并转为 Markdown，保留表格结构"""
    try:
        book = epub.read_epub(input_epub_path)
        
        # 配置 html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0  # 不自动换行
        h.protect_links = True
        
        md_content = []
        
        for item in book.get_items():
            if item.get_type() == 9:  # ITEM_DOCUMENT
                html_content = item.get_content().decode('utf-8', errors='ignore')
                markdown = h.handle(html_content)
                if markdown.strip():
                    md_content.append(markdown)
        
        full_md = '\n\n'.join(md_content)
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write(full_md)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to convert {input_epub_path}: {e}")
        return False

def convert_epub_files_to_markdown(root_path):
    """批量将 EPUB 转换为 Markdown"""
    sub_paths = os.listdir(root_path)
    tasks = []
    for sub_path in sub_paths:
        sub_root_path = os.path.join(root_path, sub_path, "epub")
        if not os.path.exists(sub_root_path):
            continue
        epub_files = [os.path.join(sub_root_path, f) 
                      for f in os.listdir(sub_root_path) if f.endswith(".epub")]
        for epub_file in epub_files:
            output_file = epub_file.replace(".epub", ".md")
            if os.path.exists(output_file):
                print(f"[INFO] Markdown already exists for {epub_file}, skipping.")
                continue
            tasks.append((epub_file, output_file))
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(convert_epub_to_markdown, src, dst): src 
                   for src, dst in tasks}
        for future in as_completed(futures):
            src = futures[future]
            if future.result():
                print(f"[INFO] Converted {src} to Markdown")

def convert_epub_to_pdf(input_epub_path: str, output_pdf_path: str):
    cmd = [
        "pandoc",
        input_epub_path,
        "-o", output_pdf_path,
        "--pdf-engine=xelatex"
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to convert {input_epub_path}: {e}")
        return False

def convert_epub_files_to_pdf(root_path):
    sub_paths = os.listdir(root_path)
    tasks = []
    for sub_path in sub_paths:
        sub_root_path = os.path.join(root_path, sub_path, "epub")
        if not os.path.exists(sub_root_path):
            continue
        epub_files = [os.path.join(sub_root_path, f) 
                      for f in os.listdir(sub_root_path) if f.endswith(".epub")]
        for epub_file in epub_files:
            if os.path.exists(epub_file.replace(".epub", ".pdf")):
                print(f"[INFO] PDF already exists for {epub_file}, skipping.")
                continue
            tasks.append((epub_file, epub_file.replace(".epub", ".pdf")))
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(convert_epub_to_pdf, src, dst): src 
                   for src, dst in tasks}
        for future in as_completed(futures):
            src = futures[future]
            if future.result():
                print(f"[INFO] Converted {src} to PDF.")


def get_chunks_from_markdown( 
        markdown_text: str,
        max_words: int = 300,
        overlap_ratio: float = 0.1
    ) -> List[str]:
    """
    将 Markdown 文档按单词数切分成多个块，适用于 RAG 向量数据库存储。
    
    Args:
        markdown_text: Markdown 格式的原文字符串
        max_words: 每个块的最大单词数，默认 300
        overlap_ratio: 块之间的重叠比例，默认 10%
    
    Returns:
        切分好的文本块列表
    """
    if not markdown_text or not markdown_text.strip():
        return []
    
    # 预处理：规范化空白字符
    text = markdown_text.strip()
    
    # 按语义边界分割（段落、标题、代码块等）
    semantic_sections = _split_by_semantic_boundaries(text)
    
    # 计算重叠单词数
    overlap_words = int(max_words * overlap_ratio)
    
    chunks = []
    current_chunk_words = []
    current_word_count = 0
    
    for section in semantic_sections:
        section_words = _tokenize(section)
        section_word_count = len(section_words)
        
        # 如果单个 section 超过 max_words，需要进一步切分
        if section_word_count > max_words:
            # 先保存当前累积的 chunk
            if current_chunk_words:
                chunks.append(' '.join(current_chunk_words))
                # 保留重叠部分
                current_chunk_words = current_chunk_words[-overlap_words:] if overlap_words > 0 else []
                current_word_count = len(current_chunk_words)
            
            # 切分大型 section
            sub_chunks = _split_large_section(section, max_words, overlap_words)
            chunks.extend(sub_chunks[:-1])  # 添加除最后一个外的所有块
            
            # 最后一个块用于继续累积
            if sub_chunks:
                last_chunk_words = _tokenize(sub_chunks[-1])
                current_chunk_words = last_chunk_words
                current_word_count = len(current_chunk_words)
        
        # 如果加入后不超过限制，直接添加
        elif current_word_count + section_word_count <= max_words:
            current_chunk_words.extend(section_words)
            current_word_count += section_word_count
        
        # 如果超过限制，保存当前块并开始新块
        else:
            if current_chunk_words:
                chunks.append(' '.join(current_chunk_words))
                # 保留重叠部分
                current_chunk_words = current_chunk_words[-overlap_words:] if overlap_words > 0 else []
                current_word_count = len(current_chunk_words)
            
            current_chunk_words.extend(section_words)
            current_word_count += section_word_count
    
    # 添加最后一个块
    if current_chunk_words:
        chunks.append(' '.join(current_chunk_words))
    
    # 后处理：清理和验证
    chunks = _post_process_chunks(chunks)
    
    return chunks

def _split_by_semantic_boundaries(text: str) -> List[str]:
    """
    按语义边界分割文本（标题、段落、代码块、列表等）
    """
    sections = []
    
    # 匹配代码块、标题、段落等
    # 代码块优先保持完整
    code_block_pattern = r'```[\s\S]*?```'
    
    # 分离代码块和普通文本
    parts = re.split(f'({code_block_pattern})', text)
    
    for part in parts:
        if not part.strip():
            continue
        
        # 代码块保持完整
        if part.startswith('```'):
            sections.append(part.strip())
        else:
            # 对普通文本按段落和标题分割
            sub_parts = re.split(r'\n\s*\n|\n(?=#)', part)
            for sub in sub_parts:
                sub = sub.strip()
                if sub:
                    sections.append(sub)
    
    return sections


def _tokenize(text: str) -> List[str]:
    """
    将文本分词（按空白字符分割）
    """
    # 使用正则匹配单词（包括中文字符作为单独的token）
    words = re.findall(r'\S+', text)
    return words


def _split_large_section(section: str, max_words: int, overlap_words: int) -> List[str]:
    """
    切分超大的单个 section
    """
    words = _tokenize(section)
    chunks = []
    
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        
        # 下一个块的起始位置（考虑重叠）
        start = end - overlap_words if overlap_words > 0 else end
        
        # 防止无限循环
        if start >= len(words) - overlap_words and end >= len(words):
            break
    
    return chunks


def _post_process_chunks(chunks: List[str]) -> List[str]:
    """
    后处理：清理空白、去除过短的块
    """
    processed = []
    min_words = 10  # 最小单词数阈值
    
    for chunk in chunks:
        chunk = chunk.strip()
        chunk = re.sub(r'\s+', ' ', chunk)  # 规范化空白
        
        word_count = len(_tokenize(chunk))
        if word_count >= min_words:
            processed.append(chunk)
        elif processed and word_count > 0:
            # 过短的块合并到前一个块
            processed[-1] = processed[-1] + ' ' + chunk
    
    return processed

def convert_pdf_to_markdown_with_marker(input_pdf_path, output_md_path):
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        )
    rendered = converter(input_pdf_path)
    text, _, images = text_from_rendered(rendered)
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(text)


def convert_pdf_to_markdown(path):
    # 判断是文件还是目录
    if os.path.isfile(path):
        pdf_files = [path]
    elif os.path.isdir(path):
        pdf_files = os.listdir(path)
        pdf_files = [os.path.join(path, f) for f in pdf_files if f.endswith(".pdf")]
    else:
        raise ValueError("输入路径必须是文件或目录")
    for pdf_file in pdf_files:
        output_md_path = pdf_file.replace(".pdf", ".md")
        convert_pdf_to_markdown_with_marker(pdf_file, output_md_path)

def get_title_from_markdown(markdown_text: str) -> str:
    """
    从 Markdown 文本中提取标题（第一个标题，可能不是一级标题，也可能是二级标题等）
    """
    lines = markdown_text.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            # 提取标题文本
            title = line.lstrip('#').strip()
            return title
    return "Untitled"


if __name__ == "__main__":
    convert_epub_files_to_markdown(root_path="/data2/home/lijidong/vllm-qwen/datafiles/guowu_data/America/downloads")

    # convert_test
    # convert_pdf_to_markdown(path="/data2/home/lijidong/vllm-qwen/datafiles/guowu_data/America_2/downloaded_pdfs")


    # convert_test
    # with open("/data2/home/lijidong/vllm-qwen/datafiles/article.md", "r", encoding="utf-8") as f:
    #     markdown_text = f.read()
    # chunks = get_chunks_from_markdown(markdown_text, max_words=300, overlap_ratio=0.1)
    # for i, chunk in enumerate(chunks):
    #     print(f"--- Chunk {i+1} ({len(_tokenize(chunk))} words) ---")
    #     print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    #     break


